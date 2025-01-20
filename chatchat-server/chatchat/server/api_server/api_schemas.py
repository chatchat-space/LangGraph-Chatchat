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
    model: str, 模型名称.
    graph: str, agent 名称.
    thread_id: int, 线程 id, 用来记录单个线程的对话历史.
    temperature:
    tools: agent 可调用的工具列表.
    stream: 是否开启流式输出.
    stream_type:
        当 stream = True 时, stream_type 为 node, token, direct 中任意之一.
        当 stream = False 时, stream_type 为 None.
    """
    messages: List[ChatCompletionMessageParam]
    model: str = get_default_llm()
    graph: str = "base_agent"
    thread_id: int
    temperature: Optional[float] = Settings.model_settings.TEMPERATURE
    max_completion_tokens: Optional[Union[int, None]] = Settings.model_settings.MAX_COMPLETION_TOKENS
    tools: Optional[List[str]] = None
    stream: Optional[bool] = True
    stream_type: Optional[Literal["node", "token", "direct", None]] = "node"
    top_k: Optional[int] = Settings.kb_settings.VECTOR_SEARCH_TOP_K
    history_len: Optional[int] = Settings.model_settings.HISTORY_LEN
    score: Optional[float] = Settings.kb_settings.SCORE_THRESHOLD


class AgentChatOutput(BaseModel):
    """
    定义了 agent 对话调用的 API 返回参数.

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
