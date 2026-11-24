"""留存纠正批次领域引擎。

解决同期群回访数据迟到时缺少成批纠正的问题：

- ``RetentionCorrectionEngine`` 持有不可变的同期群成员与观察期定义，
  群体成员和观察期只能由本群的纠正改写，外群更正会被拒绝并留痕；
- 纠正批次逐条关联回访记录与所属群的观察期，同一回访并发重送按
  自然键去重，只计数一次；
- 每个批次冻结两轮留存（纠正前/纠正后），并只保留实际产生影响的
  观察期，继续引用旧版批次的调用方仍能读到当时的合法分母；
- 复盘直接读取同一批次里的两轮留存，逐群逐期计算留存差值与实际
  新增回访人数，再按各期差值决定排名。

全部逻辑只依赖标准库、保持确定性，便于服务层与测试直接复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Any, Iterable


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class CohortProfile:
    """同期群画像：成员名单与观察期一经注册即冻结，纠正不得改写分母。"""

    cohort_id: str
    cohort_name: str
    members: frozenset[str]
    observation_windows: tuple[int, ...]


@dataclass(frozen=True)
class RevisitRecord:
    """一条迟到的回访记录；``claimed_cohort_id`` 是提交方声称归属的群。"""

    record_id: str
    user_id: str
    claimed_cohort_id: str
    revisit_day: int
    source: str = ""


@dataclass(frozen=True)
class ObservationWindowStat:
    """单个观察期的留存快照（人数口径，留存率由分母即时算出）。"""

    window_day: int
    active_count: int
    member_base_count: int

    @property
    def retention_rate(self) -> float:
        if self.member_base_count <= 0:
            return 0.0
        return round(self.active_count / self.member_base_count, 4)


@dataclass(frozen=True)
class CohortRound:
    """某一群在一轮（纠正前或纠正后）的完整留存快照。"""

    cohort_id: str
    member_base_count: int
    windows: tuple[ObservationWindowStat, ...]

    def window_stat(self, window_day: int) -> ObservationWindowStat | None:
        for stat in self.windows:
            if stat.window_day == window_day:
                return stat
        return None


@dataclass(frozen=True)
class RejectedRevisit:
    """被拒绝的回访：必须留痕，并标出实际归属群以便复盘判断补证范围。"""

    record_id: str
    user_id: str
    claimed_cohort_id: str
    actual_cohort_id: str | None
    revisit_day: int
    reason: str
    source: str = ""


@dataclass(frozen=True)
class AcceptedRevisit:
    """被接受的回访：逐条挂到对应群与实际受影响的观察期上。"""

    record_id: str
    user_id: str
    cohort_id: str
    revisit_day: int
    affected_windows: tuple[int, ...]
    source: str = ""


@dataclass
class RetentionCorrectionBatch:
    """一次成批纠正：冻结两轮留存、逐条关联结果和拒绝记录。"""

    batch_id: str
    batch_no: int
    status: str = "已计算"
    before_rounds: dict[str, CohortRound] = field(default_factory=dict)
    after_rounds: dict[str, CohortRound] = field(default_factory=dict)
    accepted: list[AcceptedRevisit] = field(default_factory=list)
    rejected: list[RejectedRevisit] = field(default_factory=list)
    duplicate_records: list[RevisitRecord] = field(default_factory=list)
    involved_cohort_ids: list[str] = field(default_factory=list)
    review_status: str = "待发布"
    review: dict[str, Any] | None = None
    pending_evidence_cohorts: set[str] = field(default_factory=set)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_records)

    def to_dict(self) -> dict[str, Any]:
        """确定性序列化：服务层据此生成通用资源行与接口响应。"""
        return {
            "batch_id": self.batch_id,
            "batch_no": self.batch_no,
            "status": self.status,
            "review_status": self.review_status,
            "involved_cohort_ids": list(self.involved_cohort_ids),
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "duplicate_count": self.duplicate_count,
            "pending_evidence_cohorts": sorted(self.pending_evidence_cohorts),
            "accepted": [
                {
                    "record_id": item.record_id,
                    "user_id": item.user_id,
                    "cohort_id": item.cohort_id,
                    "revisit_day": item.revisit_day,
                    "affected_windows": list(item.affected_windows),
                    "source": item.source,
                }
                for item in self.accepted
            ],
            "rejected": [
                {
                    "record_id": item.record_id,
                    "user_id": item.user_id,
                    "claimed_cohort_id": item.claimed_cohort_id,
                    "actual_cohort_id": item.actual_cohort_id,
                    "revisit_day": item.revisit_day,
                    "reason": item.reason,
                    "source": item.source,
                }
                for item in self.rejected
            ],
            "duplicate_records": [
                {
                    "record_id": item.record_id,
                    "user_id": item.user_id,
                    "claimed_cohort_id": item.claimed_cohort_id,
                    "revisit_day": item.revisit_day,
                }
                for item in self.duplicate_records
            ],
        }


# ============================================================
# 引擎
# ============================================================

class RetentionCorrectionEngine:
    """留存纠正与复盘的内存领域引擎（确定性、单线程语义）。"""

    def __init__(self) -> None:
        self._cohorts: dict[str, CohortProfile] = {}
        # (cohort_id, user_id, revisit_day) -> 已接受的批次/记录自然键
        self._accepted_keys: dict[tuple[str, str, int], str] = {}
        # cohort_id -> [(user_id, revisit_day), ...]，仅保存去重后的有效回访
        self._revisits: dict[str, list[tuple[str, int]]] = {}
        self._batches: dict[str, RetentionCorrectionBatch] = {}
        self._batch_seq = count(1)

    # ---------- 同期群注册与查询 ----------

    def register_cohort(
        self,
        cohort_id: str,
        cohort_name: str,
        members: Iterable[str],
        observation_windows: Iterable[int],
    ) -> CohortProfile:
        """注册同期群；成员与观察期冻结，后续纠正不能改写分母。"""
        windows = tuple(sorted({int(w) for w in observation_windows}))
        if not windows:
            raise ValueError("观察期不能为空")
        if any(w <= 0 for w in windows):
            raise ValueError("观察期必须为正整数")
        profile = CohortProfile(
            cohort_id=cohort_id,
            cohort_name=cohort_name,
            members=frozenset(members),
            observation_windows=windows,
        )
        self._cohorts[cohort_id] = profile
        return profile

    @property
    def cohorts(self) -> dict[str, CohortProfile]:
        return dict(self._cohorts)

    def _require_cohort(self, cohort_id: str) -> CohortProfile:
        profile = self._cohorts.get(cohort_id)
        if profile is None:
            raise KeyError(f"同期群 {cohort_id} 未注册")
        return profile

    def query_retention(self, cohort_id: str) -> CohortRound:
        """普通留存查询：按当前已接受回访计算该群各观察期基础结果。"""
        profile = self._require_cohort(cohort_id)
        return self._snapshot_round(profile)

    def cohort_history(self, cohort_id: str) -> list[dict[str, Any]]:
        """返回引用过该群的历史批次及当时的合法分母（旧版口径可追溯）。"""
        self._require_cohort(cohort_id)
        history: list[dict[str, Any]] = []
        for batch in self._batches.values():
            if cohort_id not in batch.involved_cohort_ids:
                continue
            before = batch.before_rounds.get(cohort_id)
            after = batch.after_rounds.get(cohort_id)
            history.append({
                "batch_id": batch.batch_id,
                "batch_no": batch.batch_no,
                "status": batch.status,
                "review_status": batch.review_status,
                "member_base_count": after.member_base_count if after else None,
                "before_windows": self._round_to_dicts(before),
                "after_windows": self._round_to_dicts(after),
            })
        return history

    # ---------- 单条回访提交（并发幂等） ----------

    def submit_revisit(
        self,
        record: RevisitRecord,
    ) -> tuple[str, AcceptedRevisit | RejectedRevisit | None]:
        """提交单条回访，返回 (结果类型, 明细)。

        结果类型为 ``accepted`` / ``rejected`` / ``duplicate``。
        并发重送同一回访时自然键相同，只会被接受一次。
        """
        profile = self._cohorts.get(record.claimed_cohort_id)
        if profile is None:
            rejected = RejectedRevisit(
                record_id=record.record_id,
                user_id=record.user_id,
                claimed_cohort_id=record.claimed_cohort_id,
                actual_cohort_id=self._locate_member(record.user_id),
                revisit_day=record.revisit_day,
                reason="声称归属的同期群不存在",
                source=record.source,
            )
            return "rejected", rejected

        if record.revisit_day <= 0:
            return "rejected", RejectedRevisit(
                record_id=record.record_id,
                user_id=record.user_id,
                claimed_cohort_id=record.claimed_cohort_id,
                actual_cohort_id=record.claimed_cohort_id,
                revisit_day=record.revisit_day,
                reason="回访天数必须为正整数",
                source=record.source,
            )

        # 群体隔离：外群用户不能借纠正写入本群
        if record.user_id not in profile.members:
            return "rejected", RejectedRevisit(
                record_id=record.record_id,
                user_id=record.user_id,
                claimed_cohort_id=record.claimed_cohort_id,
                actual_cohort_id=self._locate_member(record.user_id),
                revisit_day=record.revisit_day,
                reason="用户不属于声称归属的同期群，拒绝外群更正",
                source=record.source,
            )

        if record.revisit_day > max(profile.observation_windows):
            return "rejected", RejectedRevisit(
                record_id=record.record_id,
                user_id=record.user_id,
                claimed_cohort_id=record.claimed_cohort_id,
                actual_cohort_id=record.claimed_cohort_id,
                revisit_day=record.revisit_day,
                reason="回访天数超出该群观察期范围",
                source=record.source,
            )

        natural_key = (record.claimed_cohort_id, record.user_id, record.revisit_day)
        if natural_key in self._accepted_keys:
            return "duplicate", None

        affected = tuple(w for w in profile.observation_windows if w >= record.revisit_day)
        accepted = AcceptedRevisit(
            record_id=record.record_id,
            user_id=record.user_id,
            cohort_id=record.claimed_cohort_id,
            revisit_day=record.revisit_day,
            affected_windows=affected,
            source=record.source,
        )
        self._accepted_keys[natural_key] = record.record_id
        # 回访明细存到引擎状态里，_snapshot_round 统一读取
        self._revisits.setdefault(record.claimed_cohort_id, []).append(
            (record.user_id, record.revisit_day)
        )
        return "accepted", accepted

    def _locate_member(self, user_id: str) -> str | None:
        for profile in self._cohorts.values():
            if user_id in profile.members:
                return profile.cohort_id
        return None

    # ---------- 成批纠正 ----------

    def submit_correction_batch(
        self,
        records: Iterable[RevisitRecord],
        *,
        batch_id: str | None = None,
    ) -> RetentionCorrectionBatch:
        """新增一个留存纠正批次，逐条处理并冻结纠正前后两轮留存。"""
        batch_no = next(self._batch_seq)
        batch = RetentionCorrectionBatch(
            batch_id=batch_id or f"RCB-{batch_no:04d}",
            batch_no=batch_no,
        )

        # 纠正前：冻结全部已注册群的合法分母与当前留存
        batch.before_rounds = {
            cid: self._snapshot_round(profile)
            for cid, profile in self._cohorts.items()
        }

        involved: set[str] = set()
        for record in records:
            result, detail = self.submit_revisit(record)
            if result == "accepted":
                assert isinstance(detail, AcceptedRevisit)
                batch.accepted.append(detail)
                involved.add(detail.cohort_id)
            elif result == "duplicate":
                batch.duplicate_records.append(record)
                involved.add(record.claimed_cohort_id)
            else:
                assert isinstance(detail, RejectedRevisit)
                batch.rejected.append(detail)
                # 拒绝记录留痕并纳入复盘范围，但不会把受波及群置为待补证：
                # 待补证只用于「有效回访来源部分失效」的显式标记场景。
                involved.add(detail.claimed_cohort_id)
                if detail.actual_cohort_id:
                    involved.add(detail.actual_cohort_id)

        # 纠正后：成员分母保持不变，仅回访人数可能变化
        batch.after_rounds = {
            cid: self._snapshot_round(profile)
            for cid, profile in self._cohorts.items()
        }
        # 只保留实际受影响群的轮次快照，避免无关节点膨胀
        batch.involved_cohort_ids = sorted(involved)
        batch.before_rounds = {
            cid: batch.before_rounds[cid] for cid in batch.involved_cohort_ids
        }
        batch.after_rounds = {
            cid: batch.after_rounds[cid] for cid in batch.involved_cohort_ids
        }
        # 默认所有群均已核实；来源失效的群通过 mark_pending_evidence 显式标记
        self._batches[batch.batch_id] = batch
        return batch

    def mark_pending_evidence(self, batch_id: str, cohort_ids: Iterable[str]) -> RetentionCorrectionBatch:
        """显式标记某群回访来源部分失效、仍待补证。"""
        batch = self.get_batch(batch_id)
        for cohort_id in cohort_ids:
            if cohort_id not in batch.involved_cohort_ids:
                raise KeyError(f"同期群 {cohort_id} 不在批次 {batch_id} 范围内")
            batch.pending_evidence_cohorts.add(cohort_id)
        return batch

    def get_batch(self, batch_id: str) -> RetentionCorrectionBatch:
        if batch_id not in self._batches:
            raise KeyError(f"纠正批次 {batch_id} 不存在")
        return self._batches[batch_id]

    def list_batches(self) -> list[RetentionCorrectionBatch]:
        return [self._batches[key] for key in sorted(self._batches, key=lambda b: self._batches[b].batch_no)]

    # ---------- 复盘 ----------

    def run_review(self, batch_id: str) -> dict[str, Any]:
        """从同一批次读取两轮留存，计算逐群逐期差值与新增回访人数并排名。"""
        batch = self.get_batch(batch_id)
        cohort_entries: list[dict[str, Any]] = []
        window_deltas: dict[int, list[dict[str, Any]]] = {}

        for cohort_id in batch.involved_cohort_ids:
            before = batch.before_rounds[cohort_id]
            after = batch.after_rounds[cohort_id]
            window_results: list[dict[str, Any]] = []
            for after_stat in after.windows:
                before_stat = before.window_stat(after_stat.window_day)
                before_active = before_stat.active_count if before_stat else 0
                before_rate = before_stat.retention_rate if before_stat else 0.0
                active_delta = after_stat.active_count - before_active
                rate_delta = round(after_stat.retention_rate - before_rate, 4)
                window_results.append({
                    "window_day": after_stat.window_day,
                    "member_base_count": after_stat.member_base_count,
                    "before_active_count": before_active,
                    "after_active_count": after_stat.active_count,
                    "active_delta": active_delta,
                    "before_retention_rate": before_rate,
                    "after_retention_rate": after_stat.retention_rate,
                    "retention_delta": rate_delta,
                    "affected": active_delta != 0,
                })
                if active_delta != 0 or rate_delta != 0:
                    window_deltas.setdefault(after_stat.window_day, []).append({
                        "cohort_id": cohort_id,
                        "active_delta": active_delta,
                        "retention_delta": rate_delta,
                        "member_base_count": after_stat.member_base_count,
                    })

            # 实际新增回访人数：该群被接受回访带来的去重新增（取最大观察期净增）
            max_window = max(after.windows, key=lambda w: w.window_day)
            before_max = before.window_stat(max_window.window_day)
            newly_added = (
                max_window.active_count - (before_max.active_count if before_max else 0)
            )
            # 只保留实际产生影响的观察期，供复盘聚焦差值
            affected_windows = [
                result["window_day"] for result in window_results if result["affected"]
            ]
            cohort_entries.append({
                "cohort_id": cohort_id,
                "cohort_name": self._cohorts[cohort_id].cohort_name,
                "member_base_count": after.member_base_count,
                "newly_added_revisit_users": newly_added,
                "affected_windows": affected_windows,
                "verification_status": self._verification_status(batch, cohort_id),
                "windows": window_results,
            })

        # 各期排名：只列实际发生变化的群，按差值降序、群编号升序兜底
        window_rankings: dict[int, list[dict[str, Any]]] = {}
        for window_day, rows in window_deltas.items():
            window_rankings[window_day] = sorted(
                rows,
                key=lambda row: (-row["retention_delta"], -row["active_delta"], row["cohort_id"]),
            )

        # 总排名：按本轮实际新增回访人数降序
        overall_ranking = sorted(
            (
                {
                    "cohort_id": entry["cohort_id"],
                    "cohort_name": entry["cohort_name"],
                    "newly_added_revisit_users": entry["newly_added_revisit_users"],
                    "verification_status": entry["verification_status"],
                }
                for entry in cohort_entries
                if entry["newly_added_revisit_users"] > 0
            ),
            key=lambda row: (-row["newly_added_revisit_users"], row["cohort_id"]),
        )

        review = {
            "batch_id": batch.batch_id,
            "review_status": batch.review_status,
            "ranked_cohort_ids": [row["cohort_id"] for row in overall_ranking],
            "window_rankings": {str(day): rows for day, rows in sorted(window_rankings.items())},
            "overall_ranking": overall_ranking,
            "cohorts": cohort_entries,
            "rejected_records": [self._reject_to_dict(item) for item in batch.rejected],
            "duplicate_count": batch.duplicate_count,
        }
        batch.review = review
        return review

    def publish_review(self, batch_id: str) -> dict[str, Any]:
        """值班员核对后发布复盘；发布后批次与复盘均锁定。"""
        batch = self.get_batch(batch_id)
        if batch.review is None:
            self.run_review(batch_id)
        batch.review_status = "已发布"
        batch.status = "已发布"
        assert batch.review is not None
        batch.review["review_status"] = "已发布"
        return batch.review

    # ---------- 内部辅助 ----------

    def _snapshot_round(self, profile: CohortProfile) -> CohortRound:
        revisit_days = self._revisits.get(profile.cohort_id, [])
        windows: list[ObservationWindowStat] = []
        member_count = len(profile.members)
        for window in profile.observation_windows:
            active_users = {
                uid for uid, day in revisit_days if day <= window and uid in profile.members
            }
            windows.append(ObservationWindowStat(
                window_day=window,
                active_count=len(active_users),
                member_base_count=member_count,
            ))
        return CohortRound(
            cohort_id=profile.cohort_id,
            member_base_count=member_count,
            windows=tuple(windows),
        )

    def _verification_status(self, batch: RetentionCorrectionBatch, cohort_id: str) -> str:
        if cohort_id in batch.pending_evidence_cohorts:
            return "待补证"
        return "已核实"

    @staticmethod
    def _round_to_dicts(round_: CohortRound | None) -> list[dict[str, Any]] | None:
        if round_ is None:
            return None
        return [
            {
                "window_day": stat.window_day,
                "active_count": stat.active_count,
                "member_base_count": stat.member_base_count,
                "retention_rate": stat.retention_rate,
            }
            for stat in round_.windows
        ]

    @staticmethod
    def _reject_to_dict(item: RejectedRevisit) -> dict[str, Any]:
        return {
            "record_id": item.record_id,
            "user_id": item.user_id,
            "claimed_cohort_id": item.claimed_cohort_id,
            "actual_cohort_id": item.actual_cohort_id,
            "revisit_day": item.revisit_day,
            "reason": item.reason,
            "source": item.source,
        }


# 进程级单例：服务层共享同一份纠正状态
retention_engine = RetentionCorrectionEngine()
