# import asyncio
# import json
# import uuid
# from typing import AsyncIterable
#
# from fastapi import Body
# from sse_starlette.sse import EventSourceResponse
#
# from chatchat.server.api_server.api_schemas import OpenAIChatOutput
# from chatchat.server.callback_handler.agent_callback_handler import AgentStatus
# from chatchat.server.agent.graphs_factory.graphs_registry import Response, serialize_content
# from chatchat.server.utils import (
#     MsgType,
#     get_tool,
#     get_default_graph,
#     build_logger,
#     get_graph,
#     get_history_len,
#     get_recursion_limit,
#     create_agent_models,
# )
#
# logger = build_logger()
#
#
# async def graph_chat(
#     query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
#     model: str = Body(None, description="llm", example="gpt-4o-mini"),
#     graph: str = Body(None, description="使用的 graph 名称", example="base_graph"),
#     metadata: dict = Body({}, description="附件，可能是图像或者其他功能", examples=[]),
#     conversation_id: str = Body("", description="对话框ID"),
#     message_id: str = Body(None, description="数据库消息ID"),
#     chat_model_config: dict = Body({}, description="LLM 模型配置", examples=[]),
#     tool_config: dict = Body({}, description="工具配置", examples=[]),
#     max_tokens: int = Body(None, description="LLM 最大 token 数配置", example=4096),
#     temperature: float = Body(None, description="LLM temperature 配置", example=0.01),
#     stream: bool = Body(False, description="流式输出"),
# ):
#     """Langgraph Agent 对话"""
#     async def graph_chat_iterator() -> AsyncIterable[str]:
#         # import rich  # debug
#
#         all_tools = get_tool().values()
#         tools = [tool for tool in all_tools if tool.name in tool_config]
#         try:
#             llm = create_agent_models(configs=chat_model_config,
#                                       model=model,
#                                       max_tokens=max_tokens,
#                                       temperature=temperature,
#                                       stream=stream)
#             # 检查 llm 是否创建成功
#             if llm is None:
#                 raise Exception(f"failed to create ChatOpenAI for model: {model}.")
#         except Exception as e:
#             logger.error(f"error in create ChatOpenAI: {e}")
#             yield json.dumps({"error": str(e)})
#             return
#
#         graph_name = graph or get_default_graph() or "base_graph"
#         graph_obj = get_graph(
#             name=graph_name,
#             llm=llm,
#             tools=tools,
#             history_len=get_history_len(),
#             query=query,
#             metadata=metadata,
#         )
#         if not graph_obj:
#             raise ValueError(f"Graph '{graph_name}' is not registered.")
#
#         graph_instance = graph_obj["graph_instance"]
#         input_handler = graph_obj["input_handler"]
#         event_handler = graph_obj["event_handler"]
#
#         config = {
#             "configurable": {
#                 "thread_id": conversation_id
#             },
#             "recursion_limit": get_recursion_limit()
#         }
#
#         try:
#             # todo: 因 stream_log 输出处理太过复杂, 将来考虑是否支持, 目前暂时使用 stream
#             async for _chunk in graph_instance.astream(input_handler.create_inputs(), config, stream_mode="updates"):
#                 for node, events in _chunk.items():
#                     if node == "history_manager":  # history_manager node 为内部实现, 不外显
#                         continue
#                     content = event_handler.handle_event(node=node, events=events)
#                     serialized_content = serialize_content(content)
#                     response = Response(node=node, content=serialized_content)
#
#                     # snapshot = graph_instance.get_state(config)  # debug
#                     # rich.print(snapshot)
#
#                     graph_res = OpenAIChatOutput(
#                         id=f"chat{uuid.uuid4()}",
#                         object="chat.completion.chunk",
#                         content=json.dumps(response),
#                         role="assistant",
#                         tool_calls=[],
#                         model=llm.model_name,
#                         status=AgentStatus.agent_finish,
#                         message_type=MsgType.TEXT,
#                         message_id=message_id,
#                     )
#                     yield graph_res.model_dump_json()
#         except asyncio.exceptions.CancelledError:
#             logger.warning("Streaming progress has been interrupted by user.")
#             return
#         except Exception as e:
#             logger.error(f"Error in chatgraph: {e}")
#             yield json.dumps({"error": str(e)})
#             return
#
#         # snapshot = graph_instance.get_state(config)  # debug
#         # rich.print(snapshot)
#
#     if stream:
#         return EventSourceResponse(graph_chat_iterator())
#     else:
#         ret = OpenAIChatOutput(
#             id=f"chat{uuid.uuid4()}",
#             object="chat.completion",
#             content="",
#             role="assistant",
#             finish_reason="stop",
#             tool_calls=[],
#             status=AgentStatus.agent_finish,
#             message_type=MsgType.TEXT,
#             message_id=message_id,
#         )
#
#         async for chunk in graph_chat_iterator():
#             data = json.loads(chunk)
#             if choices := data.get("choices"):
#                 ret.content += choices[0]["delta"].get("content", "")
#             ret.model = data.get('model')
#             ret.created = data.get('created')
#
#         return ret.model_dump()
