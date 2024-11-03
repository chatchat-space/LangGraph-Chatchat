import rich
import uuid
import asyncio

import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from streamlit_extras.bottom_container import bottom

from chatchat.server.agent.graphs_factory.graphs_registry import serialize_content
from chatchat.webui_pages.utils import *
from chatchat.webui_pages.dialogue.dialogue import list_graphs, list_tools
from chatchat.server.utils import (
    build_logger,
    get_config_models,
    get_config_platforms,
    get_default_llm,
    get_history_len,
    get_graph_instance,
    get_graph_event_handler,
    get_tool,
    create_agent_models,
)

logger = build_logger()


def init_conversation_id():
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())


@st.dialog("输入初始化内容", width="large")
def article_generation_init_setting():
    article_links = st.text_area("文章链接")
    image_links = st.text_area("图片链接")

    if st.button("确认"):
        st.session_state["article_links"] = article_links
        st.session_state["image_links"] = image_links
        # 将 article_generation_init_break_point 状态扭转为 True, 后续将进行 update_state 动作
        st.session_state["article_generation_init_break_point"] = True

        user_input = (f"文章链接: {article_links}\n"
                      f"图片链接: {image_links}")
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "type": "text"  # 标识为文本类型
        })

        st.rerun()


@st.dialog("开始改写文章", width="large")
def article_generation_start_setting():
    cols = st.columns(3)
    platforms = ["所有"] + list(get_config_platforms())
    platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
    llm_models = list(
        get_config_models(
            model_type="llm", platform_name=None if platform == "所有" else platform
        )
    )
    llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
    temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])
    with st.container(height=300):
        st.markdown(st.session_state["article_list"])
    prompt = st.text_area("指令(Prompt):", value="1.将上述提供的文章内容列表,各自提炼出提纲;\n"
                                                 "2.将提纲列表整合成一篇文章的提纲;\n"
                                                 "3.按照整合后的提纲, 生成一篇新的文章, 字数要求 500字左右;\n"
                                                 "4.只需要返回最后的文章内容即可.")

    if st.button("开始编写"):
        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.session_state["prompt"] = prompt
        # 将 article_generation_start_break_point 状态扭转为 True, 后续将进行 update_state 动作
        st.session_state["article_generation_start_break_point"] = True

        user_input = (f"模型: {llm_model}\n"
                      f"温度: {temperature}\n"
                      f"指令: {prompt}")
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "type": "text"  # 标识为文本类型
        })

        st.rerun()


@st.dialog("文章重写确认", width="large")
def article_generation_repeat_setting():
    cols = st.columns(3)
    platforms = ["所有"] + list(get_config_platforms())
    platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
    llm_models = list(
        get_config_models(
            model_type="llm", platform_name=None if platform == "所有" else platform
        )
    )
    llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
    temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])
    with st.container(height=300):
        st.markdown(st.session_state["article"])
    prompt = st.text_area("指令(Prompt):", value="请继续优化, 最后只需要返回文章内容.")

    if st.button("确认-需要重写"):
        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.session_state["prompt"] = prompt
        st.session_state["article_generation_repeat_break_point"] = True

        user_input = (f"模型: {llm_model}\n"
                      f"温度: {temperature}\n"
                      f"指令: {prompt}")
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "type": "text"  # 标识为文本类型
        })
        st.rerun()
    if st.button("确认-不需要重写"):
        # 如果不需要继续改写, 则固定 prompt 如下
        prompt = "不需要继续改写文章."

        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.session_state["prompt"] = prompt
        st.session_state["article_generation_repeat_break_point"] = True
        # langgraph 退出循环的判断条件
        st.session_state["is_article_generation_complete"] = True

        user_input = (f"模型: {llm_model}\n"
                      f"温度: {temperature}\n"
                      f"指令: {prompt}")
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "type": "text"  # 标识为文本类型
        })
        st.rerun()


def extract_node_and_response(data):
    # 获取第一个键值对，作为 node
    if not data:
        raise ValueError("数据为空")

    # 获取第一个键及其对应的值
    node = next(iter(data))
    response = data[node]

    return node, response


async def handle_user_input(
        graph_name: str,
        graph_input: Any,
        graph: CompiledStateGraph,
        graph_config: Dict,
):
    events = graph.astream(input=graph_input, config=graph_config, stream_mode="updates")
    if events:
        if graph_name == "article_generation":
            async for event in events:
                node, response = extract_node_and_response(event)

                # debug
                print(f"--- node: {node} ---")
                rich.print(response)

                if node == "history_manager":  # history_manager node 为内部实现, 不外显
                    continue
                if node == "article_generation_init_break_point":
                    with st.chat_message("assistant"):
                        st.write("请进行初始化设置")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "请进行初始化设置",
                            "type": "text"  # 标识为文本类型
                        })
                    article_generation_init_setting()
                    continue
                if node == "article_generation_start_break_point":
                    with st.chat_message("assistant"):
                        st.write("请开始下达指令")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "请开始下达指令",
                            "type": "text"  # 标识为文本类型
                        })
                    st.session_state["article_list"] = response["article_list"]
                    article_generation_start_setting()
                    continue
                if node == "article_generation_repeat_break_point":
                    with st.chat_message("assistant"):
                        st.write("请确认是否重写")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "请确认是否重写",
                            "type": "text"  # 标识为文本类型
                        })
                    st.session_state["article"] = response["article"]
                    article_generation_repeat_setting()
                    continue
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    with st.status(node, expanded=True) as status:
                        st.json(response, expanded=True)
                        status.update(
                            label=node, state="complete", expanded=False
                        )
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "node": node,
                        "expanded": False,
                        "type": "json"  # 标识为JSON类型
                    })
        else:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response_last = ""
                async for event in events:
                    node, response = extract_node_and_response(event)
                    # debug
                    print(f"--- node: {node} ---")
                    # rich.print(response)
                    event_handler = get_graph_event_handler(graph_name)
                    # 获取 event
                    response = event_handler.handle_event(node=node, event=response)
                    # 将 event 转化为 json
                    response = serialize_content(response)
                    rich.print(response)
                    response_last = response["content"]

                    if node == "history_manager":  # history_manager node 为内部实现, 不外显
                        continue
                    with st.status(node, expanded=True) as status:
                        st.json(response, expanded=True)
                        status.update(
                            label=node, state="complete", expanded=False
                        )
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "node": node,
                        "expanded": False,
                        "type": "json"  # 标识为JSON类型
                    })
                st.markdown(response_last)


async def update_state(graph: CompiledStateGraph, graph_config: Dict, update_message: Dict, as_node: str):
    # rich.print(update_message)  # debug

    # print("--State before update--")
    # # 使用异步函数来获取状态历史
    # state_history = []
    # async for state in graph.aget_state_history(graph_config):
    #     state_history.append(state)
    # rich.print(state_history)

    # 更新状态
    await graph.aupdate_state(config=graph_config,
                              values=update_message,
                              as_node=as_node)

    # print("--State after update--")
    # # 再次打印状态历史
    # state_history = []
    # async for state in graph.aget_state_history(graph_config):
    #     state_history.append(state)
    # rich.print(state_history)


@st.dialog("模型配置", width="large")
def llm_model_setting():
    cols = st.columns(3)
    platforms = ["所有"] + list(get_config_platforms())
    platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
    llm_models = list(
        get_config_models(
            model_type="llm", platform_name=None if platform == "所有" else platform
        )
    )
    llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
    temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])

    if st.button("确认"):
        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.rerun()


def graph_agent_page(api: ApiRequest, is_lite: bool = False):
    # 初始化会话 id
    init_conversation_id()

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
        st.session_state["temperature"] = 0.01
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = ""
    if "article_generation_init_break_point" not in st.session_state:
        st.session_state["article_generation_init_break_point"] = False
    if "article_generation_start_break_point" not in st.session_state:
        st.session_state["article_generation_start_break_point"] = False
    if "article_generation_repeat_break_point" not in st.session_state:
        st.session_state["article_generation_repeat_break_point"] = False
    if "is_article_generation_complete" not in st.session_state:
        st.session_state["is_article_generation_complete"] = False

    with st.sidebar:
        tabs_1 = st.tabs(["工具设置"])
        with tabs_1[0]:
            graph_names = list_graphs(api)
            selected_graph = st.selectbox(
                "选择工作流",
                graph_names,
                format_func=lambda x: x,
                key="selected_graph",
                help="必选，不同的工作流的后端 agent 的逻辑不同，仅支持单选"
            )

            tools_list = list_tools(api)
            # tool_names = ["None"] + list(tools_list)
            if selected_graph == "text_to_sql":
                selected_tools = st.multiselect(
                    label="选择工具",
                    options=["query_sql_data"],
                    format_func=lambda x: tools_list[x]["title"],
                    key="selected_tools",
                    default="query_sql_data",
                    help="仅可选择 SQL查询工具"
                )
            else:
                # selected_tools demo: ['search_internet', 'search_youtube']
                selected_tools = st.multiselect(
                    label="选择工具",
                    options=list(tools_list),
                    format_func=lambda x: tools_list[x]["title"],
                    key="selected_tools",
                    default="search_internet",
                    help="支持多选"
                )

            selected_tool_configs = {
                name: tool["config"]
                for name, tool in tools_list.items()
                if name in selected_tools
            }

        st.tabs(["工作流流程图"])

    selected_tools_configs = list(selected_tool_configs)

    if selected_graph == "article_generation":
        st.title("自媒体文章生成")
        with st.chat_message("assistant"):
            st.write("Hello 👋😊，我是自媒体文章生成 Agent，输入任意内容以启动工作流～")
    elif selected_graph == "text_to_sql":
        st.title("数据库查询")
        with st.chat_message("assistant"):
            st.write("Hello 👋😊，我是数据库查询机器人，输入你想查询的内容～")
    else:
        st.title("智能聊天")
        with st.chat_message("assistant"):
            st.write("Hello 👋😊，我是智能聊天机器人，试着输入任何内容和我聊天呦～（ps: 可尝试选择多种工具）")

    with bottom():
        cols = st.columns([1, 0.2, 15, 1])
        if cols[0].button(":gear:", help="模型配置"):
            llm_model_setting()
        if cols[-1].button(":wastebasket:", help="清空对话"):
            st.session_state["messages"] = []
            st.rerun()
        if selected_graph == "article_generation":
            user_input = cols[2].chat_input("请你帮我生成一篇自媒体文章")
        elif selected_graph == "text_to_sql":
            user_input = cols[2].chat_input("请你帮忙查看组织`tcs_public`的成员有哪些？")
        else:
            user_input = cols[2].chat_input("尝试输入任何内容和我聊天呦")

    # get_tool() 是所有工具的名称和对象的 dict 的列表
    all_tools = get_tool().values()
    tools = [tool for tool in all_tools if tool.name in selected_tools_configs]
    # # 为保证调用效果, 如果用户没有选择任何 tool, 就默认选择互联网搜索工具
    # if len(tools) == 0:
    #     search_internet = get_tool(name="search_internet")
    #     tools.append(search_internet)
    # # rich.print(tools)

    # 创建 llm 实例
    # todo: max_tokens 这里有问题, None 应该是不限制, 但是目前 llm 结果为 4096
    llm_model = st.session_state["llm_model"]
    llm = create_agent_models(configs=None,
                              model=llm_model,
                              max_tokens=None,
                              temperature=st.session_state["temperature"],
                              stream=True)
    rich.print(llm)

    # 创建 langgraph 实例
    graph = get_graph_instance(
        name=selected_graph,
        llm=llm,
        tools=tools,
        history_len=get_history_len(),
    )
    if not graph:
        raise ValueError(f"Graph '{selected_graph}' is not registered.")

    # langgraph 配置文件
    graph_config = {
        "configurable": {
            "thread_id": st.session_state["conversation_id"]
        },
    }

    logger.info(f"graph: '{selected_graph}', configurable: '{graph_config}'")

    # 绘制流程图并缓存
    graph_flow_image_name = f"{selected_graph}_flow_image"
    if graph_flow_image_name not in st.session_state:
        graph_png_image = graph.get_graph().draw_mermaid_png()
        st.session_state[graph_flow_image_name] = graph_png_image
    st.sidebar.image(st.session_state[graph_flow_image_name], use_column_width=True)

    # 前端存储历史消息(仅作为 st.rerun() 时的 UI 展示)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["type"] == "json":
                with st.status(message["node"], expanded=message["expanded"]) as status:
                    st.json(message["content"], expanded=message["expanded"])
                    status.update(
                        label=message["node"], state="complete", expanded=False
                    )
            elif message["type"] == "text":
                st.markdown(message["content"])

    if selected_graph == "article_generation":
        # 初始化文章和图片信息
        if "article_links" not in st.session_state:
            st.session_state["article_links"] = ""
        if "image_links" not in st.session_state:
            st.session_state["image_links"] = ""
        if "article_links_list" not in st.session_state:
            st.session_state["article_links_list"] = []
        if "image_links_list" not in st.session_state:
            st.session_state["image_links_list"] = []

    # 对话主流程
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "type": "text"  # 标识为文本类型
        })

        # Run the async function in a synchronous context
        graph_input = {"messages": [("user", user_input)]}
        asyncio.run(handle_user_input(graph_name=selected_graph,
                                      graph_input=graph_input,
                                      graph=graph,
                                      graph_config=graph_config))

    if selected_graph == "article_generation":
        # debug
        is_article_generation_init_break_point = st.session_state["article_generation_init_break_point"]
        is_article_generation_start_break_point = st.session_state["article_generation_start_break_point"]
        is_article_generation_repeat_break_point = st.session_state["article_generation_repeat_break_point"]
        logger.info(f"断点情况: \n"
                    f"article_generation_init_break_point: {str(is_article_generation_init_break_point)}\n"
                    f"article_generation_start_break_point: {str(is_article_generation_start_break_point)}\n"
                    f"article_generation_repeat_break_point: {str(is_article_generation_repeat_break_point)}")

        # 当客户传入 文章链接 和 图片链接 后, 更新 state, 并让 langgraph 继续往下走
        if st.session_state["article_generation_init_break_point"]:
            logger.info("--- article_generation_init_break_point ---")
            update_message = {
                "article_links": st.session_state["article_links"],
                "image_links": st.session_state["image_links"],
            }
            asyncio.run(update_state(
                graph=graph,
                graph_config=graph_config,
                update_message=update_message,
                as_node="article_generation_init_break_point"
            ))
            asyncio.run(handle_user_input(graph_input=None, graph=graph, graph_config=graph_config))
            # 后续不再需要进行 爬虫动作, 将 article_generation_init_break_point 状态扭转为 False
            st.session_state["article_generation_init_break_point"] = False
        if st.session_state["article_generation_start_break_point"]:
            logger.info("--- article_generation_start_break_point ---")
            update_message = {
                "llm": st.session_state["llm_model"],
                "temperature": st.session_state["temperature"],
                "user_prompt": st.session_state["prompt"],
            }
            asyncio.run(update_state(
                graph=graph,
                graph_config=graph_config,
                update_message=update_message,
                as_node="article_generation_start_break_point"
            ))
            asyncio.run(handle_user_input(graph_input=None, graph=graph, graph_config=graph_config))
            # 后续不再需要进行 爬虫动作, 将 article_generation_init_break_point 状态扭转为 False
            st.session_state["article_generation_start_break_point"] = False
        if st.session_state["article_generation_repeat_break_point"]:
            logger.info("--- article_generation_repeat_break_point ---")
            if st.session_state["is_article_generation_complete"]:
                update_message = {
                    "llm": st.session_state["llm_model"],
                    "temperature": st.session_state["temperature"],
                    "user_prompt": st.session_state["prompt"],
                    "is_article_generation_complete": True,
                }
            else:
                update_message = {
                    "llm": st.session_state["llm_model"],
                    "temperature": st.session_state["temperature"],
                    "user_prompt": st.session_state["prompt"],
                    "is_article_generation_complete": False,
                }
            asyncio.run(update_state(
                graph=graph,
                graph_config=graph_config,
                update_message=update_message,
                as_node="article_generation_repeat_break_point"
            ))
            asyncio.run(handle_user_input(graph_input=None, graph=graph, graph_config=graph_config))
            # 将 article_generation_repeat_break_point 状态扭转为 False
            st.session_state["article_generation_start_break_point"] = False
