import rich
import asyncio
import json

from sse_starlette import EventSourceResponse
from fastapi import APIRouter

from chatchat.server.agent.graphs_factory.graphs_registry import get_graph_class
from chatchat.server.api_server.api_schemas import AgentChatInput
from chatchat.server.utils import get_checkpointer, get_graph_memory_type, create_agent_models, get_tool
from chatchat.settings import Settings
from chatchat.utils import build_logger

logger = build_logger()

chat_router = APIRouter(prefix="/v1", tags=["Agent 对话接口"])


@chat_router.post("/chat/completions")
async def openai_stream_output(
        body: AgentChatInput
):
    rich.print(body)
    if body.stream:
        async def generator():
            try:
                try:
                    graph_class = get_graph_class(body.graph)
                except ValueError as e:
                    logger.error(f"Error getting graph class: {e}")
                    yield {"data": json.dumps({"error": str(e)})}
                    return

                graph_memory_type = get_graph_memory_type()
                llm = create_agent_models(configs=None,
                                          model=body.model,
                                          max_tokens=body.max_completion_tokens,
                                          temperature=body.temperature,
                                          stream=body.stream)
                rich.print(llm)
                all_tools = get_tool().values()
                tools = [tool for tool in all_tools if tool.name in body.tools]
                rich.print(tools)
                graph_config = {
                    "configurable": {
                        "thread_id": body.thread_id
                    },
                }

                if graph_memory_type == "memory":
                    checkpointer = get_checkpointer(memory_type=graph_memory_type)
                    graph_class = graph_class(llm=llm,
                                              tools=tools,
                                              history_len=Settings.model_settings.HISTORY_LEN,
                                              checkpoint=checkpointer)
                    graph = graph_class.get_graph()
                    if not graph:
                        raise ValueError(f"Graph '{graph_class}' is not registered.")
                    async for event in graph.astream(input={"messages": body.messages},
                                                     config=graph_config,
                                                     stream_mode="updates"):
                        logger.debug(f"Event: {event}")
                        yield str(event)
                elif graph_memory_type == "sqlite":
                    checkpointer = get_checkpointer(memory_type=graph_memory_type)
                    async with checkpointer as checkpointer:
                        graph_class = graph_class(llm=llm,
                                                  tools=tools,
                                                  history_len=Settings.model_settings.HISTORY_LEN,
                                                  checkpoint=checkpointer)
                        graph = graph_class.get_graph()
                        if not graph:
                            raise ValueError(f"Graph '{graph_class}' is not registered.")
                        async for event in graph.astream(input={"messages": body.messages},
                                                         config=graph_config,
                                                         stream_mode="updates"):
                            logger.debug(f"Event: {event}")
                            yield str(event)
                elif graph_memory_type == "postgres":
                    from psycopg_pool import AsyncConnectionPool
                    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                    async with AsyncConnectionPool(
                            conninfo=Settings.basic_settings.POSTGRESQL_GRAPH_DATABASE_URI,
                            max_size=Settings.basic_settings.POSTGRESQL_GRAPH_CONNECTION_POOLS_MAX_SIZE,
                            kwargs=Settings.basic_settings.POSTGRESQL_GRAPH_CONNECTION_POOLS_KWARGS,
                    ) as pool:
                        checkpointer = AsyncPostgresSaver(pool)
                        # NOTE: you need to call .setup() the first time you're using your checkpointer
                        await checkpointer.setup()
                        graph_class = graph_class(llm=llm,
                                                  tools=tools,
                                                  history_len=Settings.model_settings.HISTORY_LEN,
                                                  checkpoint=checkpointer)
                        graph = graph_class.get_graph()
                        if not graph:
                            raise ValueError(f"Graph '{graph_class}' is not registered.")
                        async for event in graph.astream(input={"messages": body.messages},
                                                         config=graph_config,
                                                         stream_mode="updates"):
                            logger.debug(f"Event: {event}")
                            yield str(event)
            except asyncio.exceptions.CancelledError:
                logger.warning("Streaming progress has been interrupted by user.")
                return
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield {"data": json.dumps({"error": str(e)})}
                return
        return EventSourceResponse(generator())
    else:
        pass
