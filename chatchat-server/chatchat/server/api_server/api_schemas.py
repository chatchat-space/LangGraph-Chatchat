from __future__ import annotations

import json
import time

from typing import Dict, List, Literal, Optional, Union
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from chatchat.settings import Settings
from chatchat.server.utils import MsgType, get_default_llm


class AgentChatInput(BaseModel):
    messages: List[ChatCompletionMessageParam]
    model: str = get_default_llm()
    graph: str
    thread_id: int
    temperature: Optional[float] = Settings.model_settings.TEMPERATURE
    max_completion_tokens: Optional[Union[int, None]] = Settings.model_settings.MAX_COMPLETION_TOKENS
    tools: Optional[List[str]] = None
    stream: Optional[bool] = True
    stream_method: Optional[Literal["streamlog", "node", "invoke"]] = "streamlog"
    top_k: Optional[int] = Settings.kb_settings.VECTOR_SEARCH_TOP_K
    history_len: Optional[int] = Settings.model_settings.HISTORY_LEN
    score: Optional[float] = Settings.kb_settings.SCORE_THRESHOLD


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
