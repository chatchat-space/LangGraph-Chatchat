import asyncio
import rich

import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from streamlit_extras.bottom_container import bottom

from chatchat.webui_pages.utils import *
from chatchat.server.agent.graphs_factory.graphs_registry import (
    list_graph_titles_by_label,
    get_graph_class_by_label_and_title,
)
from chatchat.server.utils import (
    build_logger,
    get_config_models,
    get_config_platforms,
    get_tool,
    create_agent_models,
    list_tools,
    serialize_content_to_json
)

logger = build_logger()


def extract_node_and_response(data):
    # 获取第一个键值对，作为 node
    if not data:
        raise ValueError("数据为空")

    # 获取第一个键及其对应的值
    node = next(iter(data))
    response = data[node]

    return node, response


async def handle_user_input(
        graph: CompiledStateGraph,
        graph_input: Any,
        graph_config: Dict,
        graph_class_instance: Any
):
    events = graph.astream(input=graph_input, config=graph_config, stream_mode="updates")
    if events:
        # Display assistant response in chat message container
        with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
            response_last = ""
            async for event in events:
                node, response = extract_node_and_response(event)
                # debug
                print(f"--- node: {node} ---")
                rich.print(response)

                if node == "history_manager":  # history_manager node 为内部实现, 不外显
                    continue

                # 获取 event
                response = await graph_class_instance.handle_event(node=node, event=response)
                # 将 event 转化为 json
                response = serialize_content_to_json(response)
                # print("after serialize_content response:")
                # rich.print(response)
                response_last = response["content"]

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
                    status.update(
                        label=node, state="complete", expanded=False
                    )

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


def graph_rag_page(api: ApiRequest):
    # 初始化
    init_conversation_id()
    if "selected_kb" not in st.session_state:
        st.session_state["selected_kb"] = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE
    if "kb_top_k" not in st.session_state:
        st.session_state["kb_top_k"] = Settings.kb_settings.VECTOR_SEARCH_TOP_K
    if "score_threshold" not in st.session_state:
        st.session_state["score_threshold"] = Settings.kb_settings.SCORE_THRESHOLD

    with st.sidebar:
        tabs_1 = st.tabs(["工作流设置"])
        with tabs_1[0]:
            placeholder = st.empty()

            def on_kb_change():
                st.toast(f"已加载知识库： {st.session_state.selected_kb}")

            with placeholder.container():
                rag_graph_names = list_graph_titles_by_label(label="rag")
                selected_graph = st.selectbox(
                    "选择知识库问答工作流",
                    rag_graph_names,
                    format_func=lambda x: x,
                    key="selected_graph",
                    help="必选，不同的工作流的后端 agent 的逻辑不同，仅支持单选"
                )

                kb_list = [x["kb_name"] for x in api.list_knowledge_bases()]
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )

                tools_list = list_tools()
                # tool_names = ["None"] + list(tools_list)
                # selected_tools demo: ['search_internet', 'search_youtube']
                selected_tools = st.multiselect(
                    label="选择工具",
                    options=list(tools_list),
                    format_func=lambda x: tools_list[x]["title"],
                    key="selected_tools",
                    default="search_local_knowledgebase",
                    help="支持多选"
                )
                selected_tool_configs = {
                    name: tool["config"]
                    for name, tool in tools_list.items()
                    if name in selected_tools
                }

        tabs_2 = st.tabs(["问答设置"])
        with tabs_2[0]:
            history_len = st.number_input("历史对话轮数：", 0, 20, key="history_len")
            kb_top_k = st.number_input("匹配知识条数：", 1, 20, key="kb_top_k")
            # Bge 模型会超过 1
            score_threshold = st.slider("知识匹配分数阈值：", 0.0, 2.0, step=0.01, key="score_threshold", help="分数越小匹配度越大")

        st.tabs(["工作流流程图"])

    selected_tools_configs = list(selected_tool_configs)

    st.title("知识库聊天")
    with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
        st.write("Hello 👋😊，我是智能知识库问答机器人，试着输入任何内容和我聊天呦～（ps: 可尝试切换不同知识库）")

    with bottom():
        cols = st.columns([1, 0.2, 15, 1])
        if cols[0].button(":gear:", help="模型配置"):
            llm_model_setting()
        if cols[-1].button(":wastebasket:", help="清空对话"):
            st.session_state["messages"] = []
            st.rerun()
        user_input = cols[2].chat_input("如何给 tcr 实例开启公网访问？(换行:Shift+Enter)")

    # get_tool() 是所有工具的名称和对象的 dict 的列表
    all_tools = get_tool().values()
    tools = [tool for tool in all_tools if tool.name in selected_tools_configs]

    # 创建 llm 实例
    llm_model = st.session_state["llm_model"]
    llm = create_agent_models(configs=None,
                              model=llm_model,
                              max_tokens=None,
                              temperature=st.session_state["temperature"],
                              stream=True)
    logger.info(f"Loaded llm: {llm}")

    # 创建 langgraph 实例
    graph_class = get_graph_class_by_label_and_title(label="rag", title=selected_graph)

    if graph_class.__name__ == "BaseRagGraph":
        graph_class = graph_class(llm=llm, tools=tools, history_len=history_len, knowledge_base=selected_kb,
                                  top_k=kb_top_k, score_threshold=score_threshold)
    else:
        graph_class = graph_class(llm=llm, tools=tools, history_len=history_len, knowledge_base=selected_kb,
                                  top_k=kb_top_k, score_threshold=score_threshold)

    graph = graph_class.get_graph()
    if not graph:
        raise ValueError(f"Graph '{selected_graph}' is not registered.")
    st.toast(f"已加载工作流: {selected_graph}")

    # langgraph 配置文件
    graph_config = {
        "configurable": {
            "thread_id": st.session_state["conversation_id"]
        },
    }
    logger.info(f"Loaded graph: '{selected_graph}', configurable: '{graph_config}'")

    # 绘制流程图并缓存
    graph_flow_image_name = f"{selected_graph}_flow_image"
    if graph_flow_image_name not in st.session_state:
        graph_png_image = graph.get_graph().draw_mermaid_png()
        st.session_state[graph_flow_image_name] = graph_png_image
    st.sidebar.image(st.session_state[graph_flow_image_name], use_column_width=True)

    # 前端存储历史消息(仅作为 st.rerun() 时的 UI 展示)
    # 临时列表，用于收集 assistant 的消息
    assistant_messages = []

    # 遍历 st.session_state.messages 并展示消息
    for message in st.session_state.messages:
        role = message['role']
        content = message['content']
        is_last_message = message.get('is_last_message', False)

        if role == 'user':
            # 展示 user 消息
            with st.chat_message("user"):
                st.markdown(content)
        elif role == 'assistant':
            # 收集 assistant 消息
            assistant_messages.append(message)
            # 如果是最后一条 assistant 消息，立即展示
            if is_last_message:
                with st.chat_message(name="assistant", avatar=st.session_state["assistant_avatar"]):
                    for msg in assistant_messages:
                        if msg['is_last_message']:
                            st.markdown(msg['content'])
                        else:
                            with st.status(msg['node'], expanded=True) as status:
                                st.json(msg['content'], expanded=True)
                                status.update(
                                    label=msg['node'], state="complete", expanded=False
                                )
                # 清空临时列表
                assistant_messages = []

    # 对话主流程
    if user_input:
        st.session_state.messages.append(create_chat_message(
            role="user",
            content=user_input,
            node=None,
            expanded=None,
            type="text",
            is_last_message=True
        ))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run the async function in a synchronous context
        graph_input = {"messages": [("user", user_input)]}
        asyncio.run(handle_user_input(graph=graph, graph_input=graph_input, graph_config=graph_config, graph_class_instance=graph_class))
        st.rerun()  # Clear stale containers
