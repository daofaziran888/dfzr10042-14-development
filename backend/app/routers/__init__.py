"""业务路由汇总：所有模块共用一套资源路由，由领域服务完成分派。"""
from __future__ import annotations

from app.routers import resource, retention_correction

# 留存纠正专用路由含固定路径，必须排在通用资源路由之前注册，
# 否则 /api/retention-correction/... 会被 /api/{module} 抢先匹配。
ROUTERS = [retention_correction, resource]
