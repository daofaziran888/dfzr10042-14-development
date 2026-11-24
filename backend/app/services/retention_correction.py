"""留存纠正批次领域服务。

通用资源骨架负责列表/登记/动作；本服务额外暴露成批纠正、普通留存查询、
复盘比较与发布的业务入口，并把每个批次镜像到内存仓库，保证列表与导出
仍走统一资源路由。
"""
from __future__ import annotations

from typing import Any

from app.retention_correction import (
    RetentionCorrectionBatch,
    RetentionCorrectionEngine,
    RevisitRecord,
    retention_engine,
)
from app.services.base import BaseModuleService
from app.store import store


class RetentionCorrectionService(BaseModuleService):
    MODULE = "retention_correction"
    ENTITY = "留存纠正批次"
    OPERATION_TERM = "纠正"
    LIST_FIELDS = [
        "批次编号",
        "涉及群组",
        "接受回访",
        "拒绝记录",
        "重复提交",
        "待补证群组",
        "复盘状态",
    ]
    REQUIRED_FIELDS = ["批次编号"]
    STATUS_ORDER = ["已计算", "已发布"]
    ACTION_RULES = {"执行复盘": "已计算", "发布复盘": "已发布"}
    NEGATIVE_ACTIONS = []

    def __init__(self, engine: RetentionCorrectionEngine | None = None) -> None:
        self.engine = engine or retention_engine

    # ---------- 同期群与普通留存查询 ----------

    def register_cohort(
        self,
        cohort_id: str,
        cohort_name: str,
        members: list[str],
        observation_windows: list[int],
    ) -> dict[str, Any]:
        profile = self.engine.register_cohort(
            cohort_id, cohort_name, members, observation_windows
        )
        return {
            "cohort_id": profile.cohort_id,
            "cohort_name": profile.cohort_name,
            "member_base_count": len(profile.members),
            "observation_windows": list(profile.observation_windows),
        }

    def query_retention(self, cohort_id: str) -> dict[str, Any]:
        """普通留存查询：验证每群各观察期的基础结果。"""
        round_ = self.engine.query_retention(cohort_id)
        return {
            "cohort_id": round_.cohort_id,
            "member_base_count": round_.member_base_count,
            "windows": [
                {
                    "window_day": stat.window_day,
                    "active_count": stat.active_count,
                    "member_base_count": stat.member_base_count,
                    "retention_rate": stat.retention_rate,
                }
                for stat in round_.windows
            ],
        }

    def cohort_history(self, cohort_id: str) -> list[dict[str, Any]]:
        return self.engine.cohort_history(cohort_id)

    # ---------- 成批纠正 ----------

    def submit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        records = [self._build_record(item) for item in payload.get("records", [])]
        missing = [item for item in records if item is None]
        if missing:
            return {"ok": False, "message": "回访记录缺少 user_id、claimed_cohort_id 或 revisit_day"}
        batch = self.engine.submit_correction_batch(
            [item for item in records if item is not None],
            batch_id=payload.get("batch_id"),
        )
        for cohort_id in payload.get("pending_evidence_cohorts", []):
            if cohort_id in batch.involved_cohort_ids:
                batch.pending_evidence_cohorts.add(cohort_id)
        self._mirror_batch(batch)
        return {"ok": True, "message": "留存纠正批次已计算", "batch": batch.to_dict()}

    def review_batch(self, batch_id: str) -> dict[str, Any]:
        review = self.engine.run_review(batch_id)
        batch = self.engine.get_batch(batch_id)
        self._mirror_batch(batch)
        return review

    def publish_batch(self, batch_id: str) -> dict[str, Any]:
        review = self.engine.publish_review(batch_id)
        batch = self.engine.get_batch(batch_id)
        self._mirror_batch(batch)
        return review

    # ---------- 通用动作入口 ----------

    def run_action(self, entry_id: int, action: str) -> tuple[dict[str, Any] | None, str]:
        row = store.find(self.MODULE, entry_id)
        if row is None:
            return None, f"{self.ENTITY} {entry_id} 不存在或已归档"
        if action not in self.ACTION_RULES:
            return None, f"动作「{action}」不属于{self.MODULE}可执行范围"
        batch = self.engine.get_batch(row["批次编号"])
        if action == "执行复盘":
            self.review_batch(batch.batch_id)
            message = "复盘比较已生成，等待值班员发布"
        else:
            if batch.review is None:
                return None, "请先执行复盘再发布"
            self.publish_batch(batch.batch_id)
            message = "复盘结果已发布并锁定"
        refreshed = store.find(self.MODULE, entry_id)
        return refreshed, message

    # ---------- 内部辅助 ----------

    @staticmethod
    def _build_record(item: dict[str, Any]) -> RevisitRecord | None:
        try:
            user_id = str(item["user_id"]).strip()
            claimed = str(item["claimed_cohort_id"]).strip()
            revisit_day = int(item["revisit_day"])
        except (KeyError, TypeError, ValueError):
            return None
        if not user_id or not claimed:
            return None
        return RevisitRecord(
            record_id=str(item.get("record_id") or f"{claimed}:{user_id}:{revisit_day}"),
            user_id=user_id,
            claimed_cohort_id=claimed,
            revisit_day=revisit_day,
            source=str(item.get("source") or ""),
        )

    def _mirror_batch(self, batch: RetentionCorrectionBatch) -> dict[str, Any]:
        rows = store.rows(self.MODULE)
        payload = batch.to_dict()
        values = {
            "批次编号": payload["batch_id"],
            "涉及群组": "、".join(payload["involved_cohort_ids"]),
            "接受回访": payload["accepted_count"],
            "拒绝记录": payload["rejected_count"],
            "重复提交": payload["duplicate_count"],
            "待补证群组": "、".join(payload["pending_evidence_cohorts"]),
            "复盘状态": payload["review_status"],
        }
        for row in rows:
            if row.get("批次编号") == batch.batch_id:
                row.update(values)
                row["status"] = payload["status"]
                row["pending"] = payload["review_status"] != "已发布"
                row["abnormal"] = payload["rejected_count"] > 0 or bool(
                    payload["pending_evidence_cohorts"]
                )
                return row
        entry: dict[str, Any] = {
            "id": max((int(row.get("id", 0)) for row in rows), default=0) + 1,
            **values,
            "status": payload["status"],
            "pending": payload["review_status"] != "已发布",
            "abnormal": payload["rejected_count"] > 0
            or bool(payload["pending_evidence_cohorts"]),
        }
        rows.append(entry)
        return entry
