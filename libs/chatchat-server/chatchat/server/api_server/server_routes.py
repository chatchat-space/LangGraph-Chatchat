from fastapi import APIRouter

from chatchat.settings import Settings
from chatchat.server.utils import get_server_configs

server_router = APIRouter(prefix="/server", tags=["Server State"])

available_template_types = list(Settings.prompt_settings.model_fields.keys())

# 服务器相关接口
server_router.post(
    "/configs",
    summary="获取服务器原始配置信息",
)(get_server_configs)
