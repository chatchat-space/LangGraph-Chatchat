import os
import requests
import httpx
import openai

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import streamlit
from fastapi import FastAPI
from langchain.tools import BaseTool
from langchain_core.embeddings import Embeddings
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langgraph.checkpoint.postgres import PostgresSaver
from memoization import cached, CachingAlgorithmFlag

from chatchat.settings import Settings, XF_MODELS_TYPES
from chatchat.utils import build_logger

logger = build_logger()


def get_base_url(url):
    parsed_url = urlparse(url)  # 解析url
    base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_url)  # 格式化基础url
    return base_url.rstrip('/')


def get_config_platforms() -> Dict[str, Dict]:
    """
    获取配置的模型平台，会将 pydantic model 转换为字典。
    """
    platforms = [m.model_dump() for m in Settings.model_settings.MODEL_PLATFORMS]
    return {m["platform_name"]: m for m in platforms}


@cached(max_size=10, ttl=60, algorithm=CachingAlgorithmFlag.LRU)
def detect_xf_models(xf_url: str) -> Dict[str, List[str]]:
    '''
    use cache for xinference model detecting to avoid:
    - too many requests in short intervals
    - multiple requests to one platform for every model
    the cache will be invalidated after one minute
    '''
    xf_model_type_maps = {
        "llm_models": lambda xf_models: [k for k, v in xf_models.items()
                                        if "LLM" == v["model_type"]
                                        and "vision" not in v["model_ability"]],
        "embed_models": lambda xf_models: [k for k, v in xf_models.items()
                                        if "embedding" == v["model_type"]],
        "text2image_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if "image" == v["model_type"]],
        "image2image_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if "image" == v["model_type"]],
        "image2text_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if "LLM" == v["model_type"]
                                                and "vision" in v["model_ability"]],
        "rerank_models": lambda xf_models: [k for k, v in xf_models.items()
                                            if "rerank" == v["model_type"]],
        "speech2text_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if v.get(list(XF_MODELS_TYPES["speech2text"].keys())[0])
                                                in XF_MODELS_TYPES["speech2text"].values()],
        "text2speech_models": lambda xf_models: [k for k, v in xf_models.items()
                                                if v.get(list(XF_MODELS_TYPES["text2speech"].keys())[0])
                                                in XF_MODELS_TYPES["text2speech"].values()],
    }
    models = {}
    try:
        from xinference_client import RESTfulClient as Client
        xf_client = Client(xf_url)
        xf_models = xf_client.list_models()
        for m_type, filter in xf_model_type_maps.items():
            models[m_type] = filter(xf_models)
    except ImportError:
        logger.warning('auto_detect_model needs xinference-client installed. '
                        'Please try "pip install xinference-client". ')
    except requests.exceptions.ConnectionError:
        logger.warning(f"cannot connect to xinference host: {xf_url}, please check your configuration.")
    except Exception as e:
        logger.warning(f"error when connect to xinference server({xf_url}): {e}")
    return models


def get_config_models(
        model_name: str = None,
        model_type: Optional[Literal[
            "llm", "embed", "text2image", "image2image", "image2text", "rerank", "speech2text", "text2speech"
        ]] = None,
        platform_name: str = None,
) -> Dict[str, Dict]:
    """
    获取配置的模型列表，返回值为:
    {model_name: {
        "platform_name": xx,
        "platform_type": xx,
        "model_type": xx,
        "model_name": xx,
        "api_base_url": xx,
        "api_key": xx,
        "api_proxy": xx,
    }}
    """
    result = {}
    if model_type is None:
        model_types = [
            "llm_models",
            "embed_models",
            "text2image_models",
            "image2image_models",
            "image2text_models",
            "rerank_models",
            "speech2text_models",
            "text2speech_models",
        ]
    else:
        model_types = [f"{model_type}_models"]

    for m in list(get_config_platforms().values()):
        if platform_name is not None and platform_name != m.get("platform_name"):
            continue

        if m.get("auto_detect_model"):
            if not m.get("platform_type") == "xinference":  # TODO：当前仅支持 xf 自动检测模型
                logger.warning(f"auto_detect_model not supported for {m.get('platform_type')} yet")
                continue
            xf_url = get_base_url(m.get("api_base_url"))
            xf_models = detect_xf_models(xf_url)
            for m_type in model_types:
                # if m.get(m_type) != "auto":
                #     continue
                m[m_type] = xf_models.get(m_type, [])

        for m_type in model_types:
            models = m.get(m_type, [])
            if models == "auto":
                logger.warning("you should not set `auto` without auto_detect_model=True")
                continue
            elif not models:
                continue
            for m_name in models:
                if model_name is None or model_name == m_name:
                    result[m_name] = {
                        "platform_name": m.get("platform_name"),
                        "platform_type": m.get("platform_type"),
                        "model_type": m_type.split("_")[0],
                        "model_name": m_name,
                        "api_base_url": m.get("api_base_url"),
                        "api_key": m.get("api_key"),
                        "api_proxy": m.get("api_proxy"),
                    }
    return result


def get_model_info(
        model_name: str = None, platform_name: str = None, multiple: bool = False
) -> Dict:
    """
    获取配置的模型信息，主要是 api_base_url, api_key
    如果指定 multiple=True，则返回所有重名模型；否则仅返回第一个
    """
    result = get_config_models(model_name=model_name, platform_name=platform_name)
    if len(result) > 0:
        if multiple:
            return result
        else:
            return list(result.values())[0]
    else:
        return {}


def get_default_llm():
    available_llms = list(get_config_models(model_type="llm").keys())
    if Settings.model_settings.DEFAULT_LLM_MODEL in available_llms:
        return Settings.model_settings.DEFAULT_LLM_MODEL
    else:
        logger.warning(f"default llm model {Settings.model_settings.DEFAULT_LLM_MODEL} is not found in available llms, "
                       f"using {available_llms[0]} instead")
        return available_llms[0]


def get_default_embedding():
    available_embeddings = list(get_config_models(model_type="embed").keys())
    if Settings.model_settings.DEFAULT_EMBEDDING_MODEL in available_embeddings:
        return Settings.model_settings.DEFAULT_EMBEDDING_MODEL
    else:
        logger.warning(f"default embedding model {Settings.model_settings.DEFAULT_EMBEDDING_MODEL} is not found in "
                       f"available embeddings, using {available_embeddings[0]} instead")
        return available_embeddings[0]


def get_history_len() -> int:
    return (Settings.model_settings.HISTORY_LEN or
            Settings.model_settings.LLM_MODEL_CONFIG["action_model"]["history_len"])


def get_ChatOpenAI(
        model_name: str = get_default_llm(),
        temperature: float = Settings.model_settings.TEMPERATURE,
        max_tokens: int = Settings.model_settings.MAX_TOKENS,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        local_wrap: bool = False,  # use local wrapped api
        **kwargs: Any,
) -> ChatOpenAI:
    model_info = get_model_info(model_name)
    if max_tokens == 'None':
        max_tokens = None
    params = dict(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    # remove paramters with None value to avoid openai validation error
    for k in list(params):
        if params[k] is None:
            params.pop(k)

    try:
        if local_wrap:
            params.update(
                openai_api_base=f"{api_address()}/v1",
                openai_api_key="EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        model = ChatOpenAI(**params)
    except Exception as e:
        logger.exception(f"failed to create ChatOpenAI for model: {model_name}.")
        model = None
    return model


def get_Embeddings(
    embed_model: str = None,
    local_wrap: bool = False,  # use local wrapped api
) -> Embeddings:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_openai import OpenAIEmbeddings

    from chatchat.server.localai_embeddings import (
        LocalAIEmbeddings,
    )

    embed_model = embed_model or get_default_embedding()
    model_info = get_model_info(model_name=embed_model)
    params = dict(model=embed_model)
    try:
        if local_wrap:
            params.update(
                openai_api_base=f"{api_address()}/v1",
                openai_api_key="EMPTY",
            )
        else:
            params.update(
                openai_api_base=model_info.get("api_base_url"),
                openai_api_key=model_info.get("api_key"),
                openai_proxy=model_info.get("api_proxy"),
            )
        if model_info.get("platform_type") == "openai":
            return OpenAIEmbeddings(**params)
        elif model_info.get("platform_type") == "ollama":
            return OllamaEmbeddings(
                base_url=model_info.get("api_base_url").replace("/v1", ""),
                model=embed_model,
            )
        else:
            return LocalAIEmbeddings(**params)
    except Exception as e:
        logger.exception(f"failed to create Embeddings for model: {embed_model}.")


def check_embed_model(embed_model: str = None) -> Tuple[bool, str]:
    '''
    check weather embed_model accessable, use default embed model if None
    '''
    embed_model = embed_model or get_default_embedding()
    embeddings = get_Embeddings(embed_model=embed_model)
    try:
        embeddings.embed_query("this is a test")
        return True, ""
    except Exception as e:
        msg = f"failed to access embed model '{embed_model}': {e}"
        logger.error(msg)
        return False, msg


def get_OpenAIClient(
        platform_name: str = None,
        model_name: str = None,
        is_async: bool = True,
) -> Union[openai.Client, openai.AsyncClient]:
    """
    construct an openai Client for specified platform or model
    """
    if platform_name is None:
        platform_info = get_model_info(
            model_name=model_name, platform_name=platform_name
        )
        if platform_info is None:
            raise RuntimeError(
                f"cannot find configured platform for model: {model_name}"
            )
        platform_name = platform_info.get("platform_name")
    platform_info = get_config_platforms().get(platform_name)
    assert platform_info, f"cannot find configured platform: {platform_name}"
    params = {
        "base_url": platform_info.get("api_base_url"),
        "api_key": platform_info.get("api_key"),
    }
    httpx_params = {}
    if api_proxy := platform_info.get("api_proxy"):
        httpx_params = {
            "proxies": api_proxy,
            "transport": httpx.HTTPTransport(local_address="0.0.0.0"),
        }

    if is_async:
        if httpx_params:
            params["http_client"] = httpx.AsyncClient(**httpx_params)
        return openai.AsyncClient(**params)
    else:
        if httpx_params:
            params["http_client"] = httpx.Client(**httpx_params)
        return openai.Client(**params)


class MsgType:
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Any = Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: List[Any] = Field(..., description="List of data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    question: str = Field(..., description="Question text")
    response: str = Field(..., description="Response text")
    history: List[List[str]] = Field(..., description="History text")
    source_documents: List[str] = Field(
        ..., description="List of source documents and their scores"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n"
                            "2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n"
                            "3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n"
                            "4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n"
                            "5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n"
                            "6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，"
                        "由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t"
                    "( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "api_server" / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        """
        remove original route from app
        """
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )


def api_address(is_public: bool = False) -> str:
    '''
    允许用户在 basic_settings.API_SERVER 中配置 public_host, public_port
    以便使用云服务器或反向代理时生成正确的公网 API 地址（如知识库文档下载链接）
    '''
    from chatchat.settings import Settings

    server = Settings.basic_settings.API_SERVER
    if is_public:
        host = server.get("public_host", "127.0.0.1")
        port = server.get("public_port", "7861")
    else:
        host = server.get("host", "127.0.0.1")
        port = server.get("port", "7861")
        if host == "0.0.0.0":
            host = "127.0.0.1"
    return f"http://{host}:{port}"


def webui_address() -> str:
    from chatchat.settings import Settings

    host = Settings.basic_settings.WEBUI_SERVER["host"]
    port = Settings.basic_settings.WEBUI_SERVER["port"]
    return f"http://{host}:{port}"


def get_prompt_template(type: str, name: str) -> Optional[str]:
    """
    从prompt_config中加载模板内容
    type: 对应于 model_settings.llm_model_config 模型类别其中的一种，以及 "rag"，如果有新功能，应该进行加入。
    """

    from chatchat.settings import Settings

    return Settings.prompt_settings.model_dump().get(type, {}).get(name)


def set_httpx_config(
        timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
        unused_proxies: List[str] = [],
):
    """
    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    将本项目相关服务加入无代理列表，避免fastchat的服务器请求错误。(windows下无效)
    对于chatgpt等在线API，如要使用代理需要手动配置。搜索引擎的代理如何处置还需考虑。
    """

    import os

    import httpx

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 在进程范围内设置系统级代理
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [
        x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()
    ]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    def _get_proxies():
        return proxies

    import urllib.request

    urllib.request.getproxies = _get_proxies


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    """
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    """
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            tasks.append(pool.submit(func, **kwargs))

        for obj in as_completed(tasks):
            try:
                yield obj.result()
            except Exception as e:
                logger.exception(f"error in sub thread: {e}")


def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT,
        unused_proxies: List[str] = [],
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    """
    helper to get httpx client with default proxies that bypass local addesses.
    """
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    for x in unused_proxies:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update(
        {
            "http://": (
                os.environ.get("http_proxy")
                if os.environ.get("http_proxy")
                   and len(os.environ.get("http_proxy").strip())
                else None
            ),
            "https://": (
                os.environ.get("https_proxy")
                if os.environ.get("https_proxy")
                   and len(os.environ.get("https_proxy").strip())
                else None
            ),
            "all://": (
                os.environ.get("all_proxy")
                if os.environ.get("all_proxy")
                   and len(os.environ.get("all_proxy").strip())
                else None
            ),
        }
    )
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update(
                {"all://" + host: None}
            )  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


def get_server_configs() -> Dict:
    """
    获取configs中的原始配置项，供前端使用
    """
    _custom = {
        "api_address": api_address(),
    }

    return {**{k: v for k, v in locals().items() if k[0] != "_"}, **_custom}


def get_temp_dir(id: str = None) -> Tuple[str, str]:
    """
    创建一个临时目录，返回（路径，文件夹名称）
    """
    import uuid

    from chatchat.settings import Settings

    if id is not None:  # 如果指定的临时目录已存在，直接返回
        path = os.path.join(Settings.basic_settings.BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    id = uuid.uuid4().hex
    path = os.path.join(Settings.basic_settings.BASE_TEMP_DIR, id)
    os.mkdir(path)
    return path, id


# 动态更新知识库信息
def update_search_local_knowledgebase_tool():
    import re

    from chatchat.server.agent.tools_factory import tools_registry
    from chatchat.server.db.repository.knowledge_base_repository import list_kbs_from_db

    kbs = list_kbs_from_db()
    template = "Use local knowledgebase from one or more of these:\n{KB_info}\n to get information，Only local data on this knowledge use this tool. The 'database' should be one of the above [{key}]."
    KB_info_str = "\n".join([f"{kb.kb_name}: {kb.kb_info}" for kb in kbs])
    KB_name_info_str = "\n".join([f"{kb.kb_name}" for kb in kbs])
    template_knowledge = template.format(KB_info=KB_info_str, key=KB_name_info_str)

    search_local_knowledgebase_tool = tools_registry._TOOLS_REGISTRY.get(
        "search_local_knowledgebase"
    )
    if search_local_knowledgebase_tool:
        search_local_knowledgebase_tool.description = " ".join(
            re.split(r"\n+\s*", template_knowledge)
        )
        search_local_knowledgebase_tool.args["database"]["choices"] = [
            kb.kb_name for kb in kbs
        ]


def get_tool(name: str = None) -> Union[BaseTool, Dict[str, BaseTool]]:
    import importlib

    from chatchat.server.agent import tools_factory

    importlib.reload(tools_factory)

    from chatchat.server.agent.tools_factory import tools_registry

    update_search_local_knowledgebase_tool()
    if name is None:
        return tools_registry._TOOLS_REGISTRY
    else:
        return tools_registry._TOOLS_REGISTRY.get(name)


def list_tools():
    from chatchat.server.agent.tools_factory import tools_registry

    data = {
        t.name: {
            "name": t.name,
            "title": getattr(t, "_title", "".join([x.capitalize() for x in t.name.split("_")])),  # 动态获取 title
            "description": t.description,
            "args": t.args_schema.schema() if t.args_schema else {},
            "config": {},  # Assuming get_tool_config is not defined, you can replace it with actual config retrieval logic
        }
        for t in tools_registry._TOOLS_REGISTRY.values()
    }

    return data


def get_tool_config(name: str = None) -> Dict:
    from chatchat.settings import Settings

    if name is None:
        return Settings.tool_settings.model_dump()
    else:
        return Settings.tool_settings.model_dump().get(name, {})


# def set_graph_memory(memory_type: Literal["memory", "sqlite", "postgres", None] = None):
#     import sqlalchemy as sa
#     global _AGENT_MEMORY  # 声明使用全局 memory
#
#     if memory_type is None:
#         memory_type = Settings.tool_settings.GRAPH_MEMORY_TYPE
#
#     if hasattr(_AGENT_MEMORY, "conn"):
#         _AGENT_MEMORY.conn.close()
#
#     if memory_type == "memory":
#         from langgraph.checkpoint.memory import MemorySaver
#         _AGENT_MEMORY = MemorySaver()
#     elif memory_type == "sqlite":
#         from langgraph.checkpoint.sqlite import SqliteSaver
#
#         engine = sa.create_engine(Settings.basic_settings.SQLALCHEMY_DATABASE_URI)
#         conn = engine.connect().connection
#         _AGENT_MEMORY = SqliteSaver(conn)
#
#     elif memory_type == "postgres":
#         from langgraph.checkpoint.postgres import PostgresSaver
#
#         engine = sa.create_engine(Settings.basic_settings.SQLALCHEMY_DATABASE_URI)
#         conn = engine.connect().connection
#         _AGENT_MEMORY = PostgresSaver(conn)


def get_st_graph_memory(memory_type: Optional[Literal["memory", "sqlite", "postgres"]] = None) -> (Union)[
    MemorySaver,
    AsyncSqliteSaver,
    # PostgresSaver
]:
    """
    获取 graph 的 memory
    """
    import sqlalchemy as sa

    if memory_type is None:
        memory_type = Settings.tool_settings.GRAPH_MEMORY_TYPE

    if memory_type == "memory":
        return MemorySaver()

    elif memory_type == "sqlite":
        import aiosqlite
        conn = aiosqlite.connect("langgraph_checkpoints.sqlite")
        return AsyncSqliteSaver(conn)

    # elif memory_type == "postgres":
    #     import sqlalchemy as sa
    #     engine = sa.create_engine(Settings.basic_settings.SQLALCHEMY_DATABASE_URI)
    #     conn = engine.connect().connection
    #     return PostgresSaver(conn)

    raise ValueError("Invalid memory_type provided. Must be 'memory', 'sqlite', or 'postgres'.")


@streamlit.cache_resource
def create_agent_models(
        configs: Any,
        model: str,
        max_tokens: Any,
        temperature: float,
        stream: Any
) -> ChatOpenAI:
    """
    为适配原先 chatchat 逻辑中创建 ChatOpenAI 的函数构造此函数.
    todo: 不要同时包含传入 configs 和不传入 configs 的情况, 评判哪种合理, 只保留一种创建 ChatOpenAI 对象的方式.
    """
    # Settings.model_settings.LLM_MODEL_CONFIG 数据结构 与 UI 传入 configs 数据结构不同, 故分开处理
    if not configs:
        configs = Settings.model_settings.LLM_MODEL_CONFIG
        agent_configs = configs.get("action_model", {})
        model = model or agent_configs["model"] or get_default_llm()
        max_tokens = max_tokens or agent_configs["max_tokens"]
        # 考虑到不同来源请求时 agent 的表现, temperature 默认最高优先级使用开发组推荐配置(LLM_MODEL_CONFIG.action_model.temperature).
        # 开发者们如有需要可以将此顺序交换.
        # temperature = agent_configs["temperature"] or temperature
        temperature = temperature or agent_configs["temperature"]
    else:
        agent_configs = configs.get("action_model", {})
        model = model or agent_configs["model"] or get_default_llm()
        for _, agent_model_config in agent_configs.items():
            max_tokens = max_tokens or agent_model_config.get("max_tokens")
            temperature = agent_model_config.get("temperature") or temperature

    return get_ChatOpenAI(model_name=model, temperature=temperature, max_tokens=max_tokens, callbacks=[],
                          streaming=stream, local_wrap=False)


def add_tools_if_not_exists(
        tools_provides: List[BaseTool],
        tools_need_append: List[BaseTool]
) -> List[BaseTool]:
    """
    为避免前端没有传入 graph 所必须的 tools, 定义此函数用来追加 graph 所必须的 tools 到前端传入的 tools 合集中.
    """
    tools_provides_list = tools_provides[:]  # 使用列表来跟踪已存在的工具
    for tool in tools_need_append:
        if tool not in tools_provides_list:  # 检查工具是否已存在
            tools_provides.append(tool)  # 追加工具
            tools_provides_list.append(tool)  # 更新列表
    return tools_provides


def serialize_content_to_json(content: Any) -> Any:
    if isinstance(content, BaseModel):
        return content.dict()
    elif isinstance(content, list):
        return [serialize_content_to_json(item) for item in content]
    elif isinstance(content, dict):
        return {key: serialize_content_to_json(value) for key, value in content.items()}
    return content


if __name__ == "__main__":
    # for debug
    print(get_default_llm())
    print(get_default_embedding())
    platforms = get_config_platforms()
    models = get_config_models()
    model_info = get_model_info(platform_name="xinference-auto")
