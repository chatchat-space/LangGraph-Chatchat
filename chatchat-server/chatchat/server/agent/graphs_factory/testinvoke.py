from typing import TypedDict, Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from chatchat.server.agent.tools_factory import search_internet
from chatchat.server.utils import create_agent_models, add_tools_if_not_exists


class State(BaseModel):
    n: int


def get_chatbot() -> CompiledStateGraph:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    llm = create_agent_models(configs=None,
                              model="Qwen2.5-72B-Instruct",
                              max_tokens=None,
                              temperature=0,
                              stream=True)

    tools = add_tools_if_not_exists(tools_provides=[], tools_need_append=[search_internet])
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile()

    return graph


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    graph = get_chatbot()
    inputs = {"role": "user", "content": "Please introduce Trump based on the Internet search results."}

    print(graph.invoke(input={"messages": inputs}, stream_mode="updates"))
