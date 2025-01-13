import asyncio
import json
import logging

from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from sse_starlette.sse import EventSourceResponse
from typing import Annotated
from typing_extensions import TypedDict

from fastapi import FastAPI, Request
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from chatchat.server.agent.tools_factory import search_internet
from chatchat.server.utils import create_agent_models, add_tools_if_not_exists

app = FastAPI()
logger = logging.getLogger("uvicorn.error")


class ClientDisconnectException(Exception):
    pass


def get_chatbot() -> CompiledStateGraph:
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    llm = create_agent_models(configs=None,
                              model="qwen2.5-instruct",
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


@app.post("/stream")
async def openai_stream_output(request: Request):
    async def generator():
        graph = get_chatbot()
        inputs = {"role": "user", "content": "Please introduce Trump based on the Internet search results."}
        try:
            async for event in graph.astream(input={"messages": inputs}, stream_mode="updates"):
                disconnected = await request.is_disconnected()
                if disconnected:
                    raise ClientDisconnectException("Client disconnected")
                yield str(event)
        except asyncio.exceptions.CancelledError:
            logger.warning("Streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield {"data": json.dumps({"error": str(e)})}
            return

    return EventSourceResponse(generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
