from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from streamlit_extras.bottom_container import bottom

from chatchat.webui_pages.utils import *
from chatchat.server.agent.graphs_factory.graphs_registry import (
    list_graph_titles_by_label,
    get_graph_class_by_label_and_title,
    Graph,
)
from chatchat.server.utils import (
    build_logger,
    get_tool,
    create_agent_models,
    list_tools,
    get_checkpointer,
)

logger = build_logger()


async def create_graph(
        graph_class: Type[Graph],
        graph_input: Any,
        graph_config: dict,
        graph_llm: ChatOpenAI,
        graph_tools: list[BaseTool],
        graph_history_len: int,
        knowledge_base: str,
        top_k: int,
        score_threshold: float
):
    if st.session_state["checkpoint_type"] == "memory":
        if "memory" not in st.session_state:
            st.session_state["memory"] = get_checkpointer()
        checkpoint = st.session_state["memory"]
        graph_class = graph_class(llm=graph_llm,
                                  tools=graph_tools,
                                  history_len=graph_history_len,
                                  checkpoint=checkpoint,
                                  knowledge_base=knowledge_base,
                                  top_k=top_k,
                                  score_threshold=score_threshold)
        graph = graph_class.get_graph()
        if not graph:
            raise ValueError(f"Graph '{graph_class}' is not registered.")
        await process_graph(graph_class=graph_class, graph=graph, graph_input=graph_input, graph_config=graph_config)
    elif st.session_state["checkpoint_type"] == "sqlite":
        checkpoint_class = get_checkpointer()
        async with checkpoint_class as checkpoint:
            graph_class = graph_class(llm=graph_llm,
                                      tools=graph_tools,
                                      history_len=graph_history_len,
                                      checkpoint=checkpoint,
                                      knowledge_base=knowledge_base,
                                      top_k=top_k,
                                      score_threshold=score_threshold)
            graph = graph_class.get_graph()
            if not graph:
                raise ValueError(f"Graph '{graph_class}' is not registered.")
            await process_graph(graph_class=graph_class, graph=graph, graph_input=graph_input, graph_config=graph_config)
    elif st.session_state["checkpoint_type"] == "postgres":
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        async with AsyncConnectionPool(
            conninfo=Settings.basic_settings.POSTGRESQL_GRAPH_DATABASE_URI,
            max_size=Settings.basic_settings.POSTGRESQL_GRAPH_CONNECTION_POOLS_MAX_SIZE,
            kwargs=Settings.basic_settings.POSTGRESQL_GRAPH_CONNECTION_POOLS_KWARGS,
        ) as pool:
            checkpoint = AsyncPostgresSaver(pool)
            # NOTE: you need to call .setup() the first time you're using your checkpointer
            await checkpoint.setup()
            graph_class = graph_class(llm=graph_llm,
                                      tools=graph_tools,
                                      history_len=graph_history_len,
                                      checkpoint=checkpoint,
                                      knowledge_base=knowledge_base,
                                      top_k=top_k,
                                      score_threshold=score_threshold)
            graph = graph_class.get_graph()
            if not graph:
                raise ValueError(f"Graph '{graph_class}' is not registered.")
            await process_graph(graph_class=graph_class, graph=graph, graph_input=graph_input, graph_config=graph_config)


async def graph_rag_page(api: ApiRequest):
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
            history_len = st.number_input("历史对话轮数", 0, 20, key="history_len")
            kb_top_k = st.number_input("匹配知识条数", 1, 20, key="kb_top_k")
            # Bge 模型会超过 1
            score_threshold = st.slider("知识匹配分数阈值", 0.0, 2.0, step=0.01, key="score_threshold", help="分数越小匹配度越大")

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
        user_input = cols[2].chat_input("尝试输入任何内容和我聊天呦 (换行:Shift+Enter)")

    # get_tool() 是所有工具的名称和对象的 dict 的列表
    all_tools = get_tool().values()
    tools = [tool for tool in all_tools if tool.name in selected_tools_configs]

    # 创建 llm 实例
    llm_model = st.session_state["llm_model"]
    llm = create_agent_models(configs=None,
                              model=llm_model,
                              max_tokens=None,
                              temperature=st.session_state["temperature"],
                              stream=st.session_state["streaming"])
    st.toast(f"已加载 LLM: {llm_model}")
    logger.info(f"Loaded llm: {llm}")

    # 创建 langgraph 实例
    graph_class = get_graph_class_by_label_and_title(label="rag", title=selected_graph)

    graph_instance = st.session_state["graph_dict"].get(selected_graph)
    if graph_instance is None:
        graph_png_image = get_img_base64(f"{selected_graph}.jpg")
        # if not graph_png_image:
        #     graph_png_image = graph.get_graph().draw_mermaid_png()
        #     logger.warning(f"The graph({selected_graph}) flowchart is not found in img, use graph.draw_mermaid_png() to get it.")
        st.session_state["graph_dict"][selected_graph] = {
            "graph_class": graph_class,
            "graph_image": graph_png_image,
        }
    st.toast(f"已加载工作流: {selected_graph}")

    # langgraph 配置文件
    graph_config = {
        "configurable": {
            "thread_id": st.session_state["conversation_id"]
        },
    }
    logger.info(f"Loaded graph: '{selected_graph}', configurable: '{graph_config}'")

    st.sidebar.image(st.session_state["graph_dict"][selected_graph]["graph_image"], use_column_width=True)

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
        await create_graph(graph_class=st.session_state["graph_dict"][selected_graph]["graph_class"],
                           graph_input=graph_input,
                           graph_config=graph_config,
                           graph_llm=llm,
                           graph_tools=tools,
                           graph_history_len=history_len,
                           knowledge_base=selected_kb,
                           top_k=kb_top_k,
                           score_threshold=score_threshold)
        st.rerun()  # Clear stale containers
