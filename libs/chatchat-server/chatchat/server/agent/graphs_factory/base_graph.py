from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import build_logger, get_st_graph_memory
from .graphs_registry import State, register_graph, Graph

logger = build_logger()


@register_graph
class BaseAgentGraph(Graph):
    name = "base_agent"
    label = "agent"
    title = "聊天机器人[纯净版]"

    def __init__(self,
                 llm: ChatOpenAI,
                 tools: list[BaseTool],
                 history_len: int):
        super().__init__(llm, tools, history_len)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def chatbot(self, state: State) -> State:
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

        messages = self.llm_with_tools.invoke(state["history"])
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)
        return state

    def get_graph(self) -> CompiledStateGraph:
        """
        description: https://langchain-ai.github.io/langgraph/tutorials/introduction/
        """
        if not isinstance(self.llm, ChatOpenAI):
            raise TypeError("llm must be an instance of ChatOpenAI")
        if not all(isinstance(tool, BaseTool) for tool in self.tools):
            raise TypeError("All items in tools must be instances of BaseTool")

        memory = get_st_graph_memory()

        graph_builder = StateGraph(State)

        tool_node = ToolNode(tools=self.tools)

        graph_builder.add_node("history_manager", self.async_history_manager)
        graph_builder.add_node("chatbot", self.chatbot)
        graph_builder.add_node("tools", tool_node)

        graph_builder.set_entry_point("history_manager")
        graph_builder.add_edge("history_manager", "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")

        graph = graph_builder.compile(checkpointer=memory)
        return graph

    @staticmethod
    async def handle_event(node: str, event: State) -> BaseMessage:
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
