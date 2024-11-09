from __future__ import annotations

from fastapi import APIRouter

from chatchat.settings import Settings
from chatchat.server.utils import BaseResponse


graph_router = APIRouter(prefix="/graphs", tags=["LangGraph"])


@graph_router.get("", response_model=BaseResponse)
async def list_graph():
    return {"data": Settings.tool_settings.SUPPORT_GRAPHS}
