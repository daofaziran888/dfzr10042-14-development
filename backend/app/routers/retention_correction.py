"""留存纠正批次专用路由。

固定路径必须先于通用资源路由注册，避免被 ``/api/{module}`` 动态段抢占。
状态流转仍只在领域服务内完成，路由层只做参数装配。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import service_for
from app.services.retention_correction import RetentionCorrectionService

router = APIRouter(prefix="/api/retention-correction", tags=["留存纠正批次"])


class CohortRegisterPayload(BaseModel):
    cohort_id: str
    cohort_name: str
    members: list[str]
    observation_windows: list[int]


class RevisitItem(BaseModel):
    record_id: str | None = None
    user_id: str
    claimed_cohort_id: str
    revisit_day: int
    source: str = ""


class CorrectionBatchPayload(BaseModel):
    batch_id: str | None = None
    records: list[RevisitItem] = Field(default_factory=list)
    pending_evidence_cohorts: list[str] = Field(default_factory=list)


def _service() -> RetentionCorrectionService:
    service = service_for("retention_correction")
    assert isinstance(service, RetentionCorrectionService)
    return service


@router.post("/cohorts")
def register_cohort(payload: CohortRegisterPayload) -> dict[str, Any]:
    """注册同期群：成员与观察期冻结，后续外群更正无法改写。"""
    try:
        return _service().register_cohort(
            payload.cohort_id,
            payload.cohort_name,
            payload.members,
            payload.observation_windows,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/cohorts/{cohort_id}/retention")
def query_retention(cohort_id: str) -> dict[str, Any]:
    """普通留存查询：验证该群各观察期当前的基础留存结果。"""
    try:
        return _service().query_retention(cohort_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cohorts/{cohort_id}/history")
def cohort_history(cohort_id: str) -> dict[str, Any]:
    """历史批次口径：继续引用旧版批次时展示当时的合法分母。"""
    try:
        return {
            "cohort_id": cohort_id,
            "batches": _service().cohort_history(cohort_id),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/batches")
def submit_batch(payload: CorrectionBatchPayload) -> dict[str, Any]:
    """新增留存纠正批次：逐条关联回访与观察期，拒绝记录原样留痕。"""
    result = _service().submit_batch(payload.model_dump())
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str) -> dict[str, Any]:
    try:
        return _service().engine.get_batch(batch_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/batches/{batch_id}/review")
def review_batch(batch_id: str) -> dict[str, Any]:
    """执行复盘：读同一批次的两轮留存，算差值、新增人数与排名。"""
    try:
        return _service().review_batch(batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/batches/{batch_id}/publish")
def publish_batch(batch_id: str) -> dict[str, Any]:
    """值班员核对后发布：发布前复盘保持「待发布」，发布后锁定。"""
    try:
        return _service().publish_batch(batch_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
