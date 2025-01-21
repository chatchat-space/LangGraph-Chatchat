from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import build_logger, get_tool, add_tools_if_not_exists
from chatchat.settings import Settings
from .graphs_registry import State, register_graph, Graph

logger = build_logger()


@register_graph
class TextToSQLGraph(Graph):
    name = "text_to_sql"
    label = "agent"
    title = "数据库查询机器人[Beta]"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int,
                 checkpoint: BaseCheckpointSaver,
                 knowledge_base: str = None,
                 top_k: int = None,
                 score_threshold: float = None):
        super().__init__(llm, tools, history_len, checkpoint)
        query_sql_data = get_tool(name="query_sql_data")
        self.tools = add_tools_if_not_exists(tools_provides=self.tools, tools_need_append=[query_sql_data])
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def sql_executor(self, state: State) -> State:
        # ToolNode 默认只将结果追加到 messages 队列中, 所以需要手动在 history 中追加 ToolMessage 结果, 否则报错如下:
        # Error code: 400 -
        # {
        #     "error": {
        #         "message": "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.",
        #         "type": "invalid_request_error",
        #         "param": "messages.[1].role",
        #         "code": null
        #     }
        # }
        if isinstance(state["messages"][-1], ToolMessage):
            state["history"].append(state["messages"][-1])

        # 下面涉及的库表信息均为测试构造, 仅做参考, 请开发者自行修改.
        sql_prompt = ChatPromptTemplate.from_template(Settings.prompt_settings.text2sql["default"])

        llm_with_tools = sql_prompt | self.llm_with_tools

        messages = llm_with_tools.invoke(state)
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)

        return state

    async def result_synthesizer(self, state: State) -> State:
        # ToolNode 默认只将结果追加到 messages 队列中, 所以需要手动在 history 中追加 ToolMessage 结果, 否则报错如下:
        # Error code: 400 -
        # {
        #     "error": {
        #         "message": "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.",
        #         "type": "invalid_request_error",
        #         "param": "messages.[1].role",
        #         "code": null
        #     }
        # }
        if isinstance(state["messages"][-1], ToolMessage):
            state["history"].append(state["messages"][-1])

        result_synthesizer_prompt = ChatPromptTemplate.from_template(
            """你是一个智能数据整合机器人, 可以将数据查询结果以 Markdown 格式的表格返回.
            对话历史记录:
            {history}
            """
        )

        llm_result_synthesizer = result_synthesizer_prompt | self.llm

        messages = llm_result_synthesizer.invoke(state)
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)

        return state

    def get_graph(self) -> CompiledStateGraph:
        """
        description: text to sql, only select
        """
        if not isinstance(self.llm, ChatOpenAI):
            raise TypeError("llm must be an instance of ChatOpenAI")
        if not all(isinstance(tool, BaseTool) for tool in self.tools):
            raise TypeError("All items in tools must be instances of BaseTool")

        graph_builder = StateGraph(State)

        tool_node = ToolNode(tools=self.tools)

        graph_builder.add_node("history_manager", self.async_history_manager)
        graph_builder.add_node("sql_executor", self.sql_executor)
        graph_builder.add_node("result_synthesizer", self.result_synthesizer)
        graph_builder.add_node("tools", tool_node)

        graph_builder.set_entry_point("history_manager")
        graph_builder.add_edge("history_manager", "sql_executor")
        graph_builder.add_conditional_edges(
            "sql_executor",
            tools_condition,
        )
        graph_builder.add_edge("tools", "result_synthesizer")
        graph_builder.add_edge("result_synthesizer", END)

        graph = graph_builder.compile(checkpointer=self.checkpoint)

        return graph

    @staticmethod
    def handle_event(node: str, event: State) -> BaseMessage:
        """
        event example:
        {
            'messages': [HumanMessage(
                            content='The youtube video of Xiao Yixian in Fights Break Sphere?',
                            id='b9c5468a-7340-425b-ae6f-2f584a961014')],
            'history': [HumanMessage(
                            content='The youtube video of Xiao Yixian in Fights Break Sphere?',
                            id='b9c5468a-7340-425b-ae6f-2f584a961014')]
        }
        """
        return event["messages"][-1]
