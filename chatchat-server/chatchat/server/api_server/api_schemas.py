from __future__ import annotations

import json
import time

from typing import Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field
from openai.types.chat import ChatCompletionMessageParam

from chatchat.settings import Settings
from chatchat.server.utils import MsgType, get_default_llm


class AgentChatInput(BaseModel):
    """
    定义了 agent 对话调用的 API 请求参数.

    messages: 必选, list, 消息列表.
    model: 必选, str, 模型名称.
    graph: 必选, str, agent 名称.
    thread_id: 必选, int, 线程 id, 用来记录单个线程的对话历史.
    history_len: 必选, int, agent 具备的短期记忆窗口长度.
    temperature: 可选, float, 调用模型时 temperature 参数值.
    max_completion_tokens: 可选, int, 调用模型时 max_completion_tokens 参数值.
    tools: 可选, list[str], agent 可调用的工具列表.
    stream: 可选, bool, 是否开启流式输出, 默认为 True.
    stream_type: 可选,
        当 stream = True 时, stream_type 为 node, token 中任意之一:
            node 指 node 级别的流式输出;
            token 指 token 级别的流式输出.
        当 stream = False 时, stream_type 为 null, 非流式输出, 直接返回最终结果.
    knowledge_base: 可选(RAG 必选), str, 检索文档时的知识库.
    top_k: 可选(RAG 必选), int, 检索文档时保留的文档数量.
    score: 可选(RAG 必选), float, 检索文档时分数阈值.
    """
    messages: List[ChatCompletionMessageParam]
    model: str = get_default_llm()
    graph: str = "base_agent"
    thread_id: int
    history_len: Optional[int] = Settings.model_settings.HISTORY_LEN
    temperature: Optional[float] = Settings.model_settings.TEMPERATURE
    max_completion_tokens: Optional[Union[int, None]] = Settings.model_settings.MAX_COMPLETION_TOKENS
    tools: Optional[List[str]] = None
    stream: Optional[bool] = True
    stream_type: Optional[Literal["node", "token", None]] = "node"
    knowledge_base: Optional[str] = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE
    top_k: Optional[int] = Settings.kb_settings.VECTOR_SEARCH_TOP_K
    score: Optional[float] = Settings.kb_settings.SCORE_THRESHOLD


class AgentChatOutput(BaseModel):
    """
    定义了 agent 对话调用的 API 返回参数.

    node: agent 的 node
    metadata: agenet 的 node 的 metadata
    messages: agent 返回结果
    """
    node: Optional[str]
    metadata: Optional[dict]
    messages: Any


class OpenAIBaseOutput(BaseModel):
    id: Optional[str] = None
    content: Optional[str] = None
    model: Optional[str] = None
    object: Literal[
        "chat.completion", "chat.completion.chunk"
    ] = "chat.completion.chunk"
    role: Literal["assistant"] = "assistant"
    finish_reason: Optional[str] = None
    created: int = Field(default_factory=lambda: int(time.time()))
    tool_calls: List[Dict] = []

    status: Optional[int] = None  # AgentStatus
    message_type: int = MsgType.TEXT
    message_id: Optional[str] = None  # id in database table
    is_ref: bool = False  # wheather show in seperated expander

    class Config:
        extra = "allow"

    def model_dump(self) -> dict:
        result = {
            "id": self.id,
            "object": self.object,
            "model": self.model,
            "created": self.created,
            "status": self.status,
            "message_type": self.message_type,
            "message_id": self.message_id,
            "is_ref": self.is_ref,
            **(self.model_extra or {}),
        }

        if self.object == "chat.completion.chunk":
            result["choices"] = [
                {
                    "delta": {
                        "content": self.content,
                        "tool_calls": self.tool_calls,
                    },
                    "role": self.role,
                }
            ]
        elif self.object == "chat.completion":
            result["choices"] = [
                {
                    "message": {
                        "role": self.role,
                        "content": self.content,
                        "finish_reason": self.finish_reason,
                        "tool_calls": self.tool_calls,
                    }
                }
            ]
        return result

    def model_dump_json(self):
        return json.dumps(self.model_dump(), ensure_ascii=False)


class OpenAIChatOutput(OpenAIBaseOutput):
    ...
