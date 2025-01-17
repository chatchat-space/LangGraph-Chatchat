import asyncio
import json
from typing import TypedDict, Annotated

import rich
from fastapi import APIRouter
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sse_starlette import EventSourceResponse

from chatchat.server.agent.tools_factory import search_internet, search_youtube
from chatchat.server.api_server.api_schemas import AgentChatInput
from chatchat.server.utils import create_agent_models
from chatchat.utils import build_logger


logger = build_logger()

chat_router = APIRouter(prefix="/v1", tags=["Agent 对话接口"])


def get_chatbot() -> CompiledStateGraph:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    llm = create_agent_models(configs=None,
                              model="hunyuan-turbo",
                              max_tokens=None,
                              temperature=0,
                              stream=True)

    tools = [search_internet, search_youtube]
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


@chat_router.post("/chat/completions")
async def openai_stream_output(
        body: AgentChatInput
):
    rich.print(body)

    async def generator():
        graph = get_chatbot()
        try:
            # async for event in graph.astream(input={"messages": inputs}, stream_mode="updates"):
            async for event in graph.astream(input={"messages": body.messages}, stream_mode="updates"):
                yield str(event)
        except asyncio.exceptions.CancelledError:
            logger.warning("Streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield {"data": json.dumps({"error": str(e)})}
            return

    return EventSourceResponse(generator())
