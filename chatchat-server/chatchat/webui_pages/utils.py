# 该文件封装了对api.py的请求，可以被不同的webui使用
# 通过ApiRequest和AsyncApiRequest支持同步/异步调用
import base64
import contextlib
import json
import os
from io import BytesIO
from pathlib import Path
from typing import *
import httpx
import uuid

import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from chatchat.server.agent.graphs_factory.graphs_registry import Graph
from chatchat.settings import Settings
from chatchat.server.utils import (
    api_address,
    get_httpx_client,
    set_httpx_config,
    get_default_embedding,
    get_default_llm, get_config_platforms, get_config_models
)
from chatchat.utils import build_logger


logger = build_logger()

set_httpx_config()


class ApiRequest:
    """
    api.py调用的封装（同步模式）,简化api调用方式
    """

    def __init__(
        self,
        base_url: str = api_address(),
        timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._use_async = False
        self._client = None

    @property
    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = get_httpx_client(
                base_url=self.base_url, use_async=self._use_async, timeout=self.timeout
            )
        return self._client

    def get(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("GET", url, params=params, **kwargs)
                else:
                    return self.client.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when get {url}: {e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                retry -= 1

    def post(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                # print(kwargs)
                if stream:
                    return self.client.stream(
                        "POST", url, data=data, json=json, **kwargs
                    )
                else:
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when post {url}: {e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                retry -= 1

    def delete(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream(
                        "DELETE", url, data=data, json=json, **kwargs
                    )
                else:
                    return self.client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when delete {url}: {e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                retry -= 1

    def _httpx_stream2generator(
        self,
        response: contextlib._GeneratorContextManager,
        as_json: bool = False,
    ):
        """
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        """

        async def ret_async(response, as_json):
            try:
                async with response as r:
                    chunk_cache = ""
                    async for chunk in r.aiter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk_cache + chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk_cache + chunk)

                                chunk_cache = ""
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                logger.error(f"{e.__class__.__name__}: {msg}")

                                if chunk.startswith("data: "):
                                    chunk_cache += chunk[6:-2]
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    chunk_cache += chunk
                                continue
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                yield {"code": 500, "msg": msg}

        def ret_sync(response, as_json):
            try:
                with response as r:
                    chunk_cache = ""
                    for chunk in r.iter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk_cache + chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk_cache + chunk)

                                chunk_cache = ""
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                logger.error(f"{e.__class__.__name__}: {msg}")

                                if chunk.startswith("data: "):
                                    chunk_cache += chunk[6:-2]
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    chunk_cache += chunk
                                continue
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                yield {"code": 500, "msg": msg}

        if self._use_async:
            return ret_async(response, as_json)
        else:
            return ret_sync(response, as_json)

    def _get_response_value(
        self,
        response: httpx.Response,
        as_json: bool = False,
        value_func: Callable = None,
    ):
        """
        转换同步或异步请求返回的响应
        `as_json`: 返回json
        `value_func`: 用户可以自定义返回值，该函数接受response或json
        """

        def to_json(r):
            try:
                return r.json()
            except Exception as e:
                msg = "API未能返回正确的JSON。" + str(e)
                logger.error(f"{e.__class__.__name__}: {msg}")
                return {"code": 500, "msg": msg, "data": None}

        if value_func is None:
            value_func = lambda r: r

        async def ret_async(response):
            if as_json:
                return value_func(to_json(await response))
            else:
                return value_func(await response)

        if self._use_async:
            return ret_async(response)
        else:
            if as_json:
                return value_func(to_json(response))
            else:
                return value_func(response)

    # 服务器信息
    def get_server_configs(self, **kwargs) -> Dict:
        response = self.post("/server/configs", **kwargs)
        return self._get_response_value(response, as_json=True)

    def get_prompt_template(
        self,
        type: str = "llm_chat",
        name: str = "default",
        **kwargs,
    ) -> str:
        data = {
            "type": type,
            "name": name,
        }
        response = self.post("/server/get_prompt_template", json=data, **kwargs)
        return self._get_response_value(response, value_func=lambda r: r.text)

    # 对话相关操作
    def chat_chat(
        self,
        query: str,
        metadata: dict,
        conversation_id: str = None,
        history_len: int = -1,
        history: List[Dict] = [],
        stream: bool = True,
        chat_model_config: Dict = None,
        tool_config: Dict = None,
        **kwargs,
    ):
        """
        对应api.py/chat/chat接口
        """
        data = {
            "query": query,
            "metadata": metadata,
            "conversation_id": conversation_id,
            "history_len": history_len,
            "history": history,
            "stream": stream,
            "chat_model_config": chat_model_config,
            "tool_config": tool_config,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post("/chat/chat", json=data, stream=True, **kwargs)
        return self._httpx_stream2generator(response, as_json=True)

    def upload_temp_docs(
        self,
        files: List[Union[str, Path, bytes]],
        knowledge_id: str = None,
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
    ):
        """
        对应api.py/knowledge_base/upload_temp_docs接口
        """

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_id": knowledge_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        response = self.post(
            "/knowledge_base/upload_temp_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def file_chat(
        self,
        query: str,
        knowledge_id: str,
        top_k: int = Settings.kb_settings.VECTOR_SEARCH_TOP_K,
        score_threshold: float = Settings.kb_settings.SCORE_THRESHOLD,
        history: List[Dict] = [],
        stream: bool = True,
        model: str = None,
        temperature: float = 0.9,
        max_tokens: int = None,
        prompt_name: str = "default",
    ):
        """
        对应api.py/chat/file_chat接口
        """
        data = {
            "query": query,
            "knowledge_id": knowledge_id,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        response = self.post(
            "/chat/file_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    # 知识库相关操作

    def list_knowledge_bases(
        self,
    ):
        """
        对应api.py/knowledge_base/list_knowledge_bases接口
        """
        response = self.get("/knowledge_base/list_knowledge_bases")
        return self._get_response_value(
            response, as_json=True, value_func=lambda r: r.get("data", [])
        )

    def create_knowledge_base(
        self,
        knowledge_base_name: str,
        vector_store_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
        embed_model: str = get_default_embedding(),
    ):
        """
        对应api.py/knowledge_base/create_knowledge_base接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }

        response = self.post(
            "/knowledge_base/create_knowledge_base",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def delete_knowledge_base(
        self,
        knowledge_base_name: str,
    ):
        """
        对应api.py/knowledge_base/delete_knowledge_base接口
        """
        response = self.post(
            "/knowledge_base/delete_knowledge_base",
            json=f"{knowledge_base_name}",
        )
        return self._get_response_value(response, as_json=True)

    def list_kb_docs(
        self,
        knowledge_base_name: str,
    ):
        """
        对应api.py/knowledge_base/list_files接口
        """
        response = self.get(
            "/knowledge_base/list_files",
            params={"knowledge_base_name": knowledge_base_name},
        )
        return self._get_response_value(
            response, as_json=True, value_func=lambda r: r.get("data", [])
        )

    def search_kb_docs(
        self,
        knowledge_base_name: str,
        query: str = "",
        top_k: int = Settings.kb_settings.VECTOR_SEARCH_TOP_K,
        score_threshold: int = Settings.kb_settings.SCORE_THRESHOLD,
        file_name: str = "",
        metadata: dict = {},
    ) -> List:
        """
        对应api.py/knowledge_base/search_docs接口
        """
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "file_name": file_name,
            "metadata": metadata,
        }

        response = self.post(
            "/knowledge_base/search_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def upload_kb_docs(
        self,
        files: List[Union[str, Path, bytes]],
        knowledge_base_name: str,
        override: bool = False,
        to_vector_store: bool = True,
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
        docs: Dict = {},
        not_refresh_vs_cache: bool = False,
    ):
        """
        对应api.py/knowledge_base/upload_docs接口
        """

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_base_name": knowledge_base_name,
            "override": override,
            "to_vector_store": to_vector_store,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)
        response = self.post(
            "/knowledge_base/upload_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def delete_kb_docs(
        self,
        knowledge_base_name: str,
        file_names: List[str],
        delete_content: bool = False,
        not_refresh_vs_cache: bool = False,
    ):
        """
        对应api.py/knowledge_base/delete_docs接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        response = self.post(
            "/knowledge_base/delete_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_info(self, knowledge_base_name, kb_info):
        """
        对应api.py/knowledge_base/update_info接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "kb_info": kb_info,
        }

        response = self.post(
            "/knowledge_base/update_info",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_docs(
        self,
        knowledge_base_name: str,
        file_names: List[str],
        override_custom_docs: bool = False,
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
        docs: Dict = {},
        not_refresh_vs_cache: bool = False,
    ):
        """
        对应api.py/knowledge_base/update_docs接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "override_custom_docs": override_custom_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)

        response = self.post(
            "/knowledge_base/update_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def recreate_vector_store(
        self,
        knowledge_base_name: str,
        allow_empty_kb: bool = True,
        vs_type: str = Settings.kb_settings.DEFAULT_VS_TYPE,
        embed_model: str = get_default_embedding(),
        chunk_size=Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
        zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
    ):
        """
        对应api.py/knowledge_base/recreate_vector_store接口
        """
        data = {
            "knowledge_base_name": knowledge_base_name,
            "allow_empty_kb": allow_empty_kb,
            "vs_type": vs_type,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }

        response = self.post(
            "/knowledge_base/recreate_vector_store",
            json=data,
            stream=True,
            timeout=None,
        )
        return self._httpx_stream2generator(response, as_json=True)

    def embed_texts(
        self,
        texts: List[str],
        embed_model: str = get_default_embedding(),
        to_query: bool = False,
    ) -> List[List[float]]:
        """
        对文本进行向量化，可选模型包括本地 embed_models 和支持 embeddings 的在线模型
        """
        data = {
            "texts": texts,
            "embed_model": embed_model,
            "to_query": to_query,
        }
        resp = self.post(
            "/other/embed_texts",
            json=data,
        )
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data")
        )

    def chat_feedback(
        self,
        message_id: str,
        score: int,
        reason: str = "",
    ) -> int:
        """
        反馈对话评价
        """
        data = {
            "message_id": message_id,
            "score": score,
            "reason": reason,
        }
        resp = self.post("/chat/feedback", json=data)
        return self._get_response_value(resp)

    def list_tools(self) -> Dict:
        """
        列出所有工具
        """
        resp = self.get("/tools")
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data", {})
        )

    def list_graphs(self) -> List[str]:
        """
        列出所有 graph
        """
        resp = self.get("/graphs")
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data", {})
        )

    def call_tool(
        self,
        name: str,
        tool_input: Dict = {},
    ):
        """
        调用工具
        """
        data = {
            "name": name,
            "tool_input": tool_input,
        }
        resp = self.post("/tools/call", json=data)
        return self._get_response_value(
            resp, as_json=True, value_func=lambda r: r.get("data")
        )


class AsyncApiRequest(ApiRequest):
    def __init__(
        self, base_url: str = api_address(), timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT
    ):
        super().__init__(base_url, timeout)
        self._use_async = True


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    """
    return error message if error occured when requests API
    """
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    """
    return error message if error occured when requests API
    """
    if (
        isinstance(data, dict)
        and key in data
        and "code" in data
        and data["code"] == 200
    ):
        return data[key]
    return ""


def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    image = f"{Settings.basic_settings.IMG_DIR}/{file_name}"
    # 读取图片
    with open(image, "rb") as f:
        buffer = BytesIO(f.read())
        base_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{base_str}"


def create_chat_message(
        role: Literal["user", "assistant"],
        content: str,
        node: Optional[str],
        expanded: Optional[bool],
        type: Literal["text", "json"],
        is_last_message: bool
):
    return {
        "role": role,
        "content": content,
        "node": node,
        "expanded": expanded,
        "type": type,
        "is_last_message": is_last_message
    }


def init_conversation_id():
    """
    公共配置初始化
    """
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())
    # 设置默认头像
    if "assistant_avatar" not in st.session_state:
        st.session_state["assistant_avatar"] = get_img_base64("chatchat_icon_blue_square_v2.png")
    # 创建 streamlit 消息缓存
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 初始化模型配置
    if "platform" not in st.session_state:
        st.session_state["platform"] = "所有"
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = get_default_llm()
        logger.info("default llm model: {}".format(st.session_state["llm_model"]))
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = Settings.model_settings.TEMPERATURE
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = ""
    if "history_len" not in st.session_state:
        st.session_state["history_len"] = Settings.model_settings.HISTORY_LEN
    if "graph_dict" not in st.session_state:
        st.session_state["graph_dict"] = {}
    if "checkpoint_type" not in st.session_state:
        st.session_state["checkpoint_type"] = Settings.tool_settings.GRAPH_MEMORY_TYPE
    if "streaming" not in st.session_state:
        st.session_state["streaming"] = False


def extract_node_and_response(data):
    # 获取第一个键值对，作为 node
    if not data:
        raise ValueError("数据为空")

    # 获取第一个键及其对应的值
    node = next(iter(data))
    response = data[node]

    return node, response


def serialize_content_to_json(content: Any) -> Any:
    if isinstance(content, BaseModel):
        return content.dict()
    elif isinstance(content, list):
        return [serialize_content_to_json(item) for item in content]
    elif isinstance(content, dict):
        return {key: serialize_content_to_json(value) for key, value in content.items()}
    return content


async def process_graph(graph_class: Graph, graph: CompiledStateGraph, graph_input: Any, graph_config: dict):
    events = graph.astream(input=graph_input, config=graph_config, stream_mode="updates")
    if events:
        # Display assistant response in chat message container
        with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
            response_last = ""
            async for event in events:
                node, response = extract_node_and_response(event)

                # debug
                # import rich
                # print(f"--- node: {node} ---")
                # rich.print(response)

                if node == "history_manager":  # history_manager node 为内部实现, 不外显
                    continue

                # 获取 event
                response = graph_class.handle_event(node=node, event=response)
                # 将 event 转化为 json
                response = serialize_content_to_json(response)

                # debug
                import rich
                print(f"--- node: {node} ---")
                rich.print(response)

                # 检查 'content' 是否在响应中(因为我们只需要 AIMessage 的内容)
                if "content" in response:
                    response_last = response["content"]
                elif "response" in response:  # plan_execute_agent
                    response_last = response["response"]
                elif "answer" in response:  # reflexion
                    response_last = response["answer"]

                # Add assistant response to chat history
                st.session_state.messages.append(create_chat_message(
                    role="assistant",
                    content=response,
                    node=node,
                    expanded=False,
                    type="json",
                    is_last_message=False
                ))
                with st.status(node, expanded=True) as status:
                    st.json(response, expanded=True)
                    status.update(label=node, state="complete", expanded=False)

            # Add assistant response_last to chat history
            st.session_state.messages.append(create_chat_message(
                role="assistant",
                content=response_last,
                node=None,
                expanded=None,
                type="text",
                is_last_message=True
            ))
            st.markdown(response_last)

        # if graph_class_instance.name == "article_generation":
        #     async for event in events:
        #         node, response = extract_node_and_response(event)
        #
        #         # debug
        #         # print(f"--- node: {node} ---")
        #         # rich.print(response)
        #
        #         if node == "history_manager":  # history_manager node 为内部实现, 不外显
        #             continue
        #         if node == "article_generation_init_break_point":
        #             with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
        #                 st.write("请进行初始化设置")
        #                 st.session_state.messages.append({
        #                     "role": "assistant",
        #                     "content": "请进行初始化设置",
        #                     "type": "text"  # 标识为文本类型
        #                 })
        #             article_generation_init_setting()
        #             continue
        #         if node == "article_generation_start_break_point":
        #             with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
        #                 st.write("请开始下达指令")
        #                 st.session_state.messages.append({
        #                     "role": "assistant",
        #                     "content": "请开始下达指令",
        #                     "type": "text"  # 标识为文本类型
        #                 })
        #             st.session_state["article_list"] = response["article_list"]
        #             article_generation_start_setting()
        #             continue
        #         if node == "article_generation_repeat_break_point":
        #             with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
        #                 st.write("请确认是否重写")
        #                 st.session_state.messages.append({
        #                     "role": "assistant",
        #                     "content": "请确认是否重写",
        #                     "type": "text"  # 标识为文本类型
        #                 })
        #             st.session_state["article"] = response["article"]
        #             article_generation_repeat_setting()
        #             continue
        #         # Display assistant response in chat message container
        #         with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
        #             with st.status(node, expanded=True) as status:
        #                 st.json(response, expanded=True)
        #                 status.update(
        #                     label=node, state="complete", expanded=False
        #                 )
        #             # Add assistant response to chat history
        #             st.session_state.messages.append({
        #                 "role": "assistant",
        #                 "content": response,
        #                 "node": node,
        #                 "expanded": False,
        #                 "type": "json"  # 标识为JSON类型
        #             })
        # else:
        #     # Display assistant response in chat message container
        #     with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
        #         response_last = ""
        #         async for event in events:
        #             node, response = extract_node_and_response(event)
        #             # debug
        #             # print(f"--- node: {node} ---")
        #             # rich.print(response)
        #
        #             if node == "history_manager":  # history_manager node 为内部实现, 不外显
        #                 continue
        #
        #             # 获取 event
        #             response = graph_class_instance.handle_event(node=node, event=response)
        #             # 将 event 转化为 json
        #             response = serialize_content_to_json(response)
        #             # rich.print(response)
        #
        #             # 检查 'content' 是否在响应中(因为我们只需要 AIMessage 的内容)
        #             if "content" in response:
        #                 response_last = response["content"]
        #             elif "response" in response:  # plan_execute_agent
        #                 response_last = response["response"]
        #             elif "answer" in response:  # reflexion
        #                 response_last = response["answer"]
        #
        #             # Add assistant response to chat history
        #             st.session_state.messages.append(create_chat_message(
        #                 role="assistant",
        #                 content=response,
        #                 node=node,
        #                 expanded=False,
        #                 type="json",
        #                 is_last_message=False
        #             ))
        #             with st.status(node, expanded=True) as status:
        #                 st.json(response, expanded=True)
        #                 status.update(label=node, state="complete", expanded=False)
        #
        #         # Add assistant response_last to chat history
        #         st.session_state.messages.append(create_chat_message(
        #             role="assistant",
        #             content=response_last,
        #             node=None,
        #             expanded=None,
        #             type="text",
        #             is_last_message=True
        #         ))
        #         st.markdown(response_last)


def check_model_supports_streaming(llm_model: str):
    """
    这里实现检查模型是否支持流式传输的逻辑
    返回 True 或 False
    """
    # todo: 需要实现更精细的"关于模型是否支持流式输出"的判断逻辑
    if llm_model == "qwen2.5-instruct":
        return False
    else:
        return True


@st.dialog("模型配置", width="large")
def llm_model_setting():
    cols = st.columns(3)
    platforms = ["所有"] + list(get_config_platforms())
    platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
    llm_models = list(get_config_models(model_type="llm", platform_name=None if platform == "所有" else platform))
    llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
    temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])
    # 检查所选模型是否支持流式传输
    supports_streaming = check_model_supports_streaming(llm_model)  # 需要实现此函数
    streaming = st.checkbox("启用流式传输(Streaming)", value=supports_streaming, help="不支持流式输出的模型勾选后会报错")

    if st.button("确认"):
        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.session_state["streaming"] = streaming
        st.rerun()


if __name__ == "__main__":
    api = ApiRequest()
    aapi = AsyncApiRequest()
