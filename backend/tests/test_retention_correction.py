"""洞察工坊 · 留存纠正批次与复盘测试

覆盖场景：
- 甲、乙同期群各两人；甲有一人第 3 天回访，乙无人回访；
- 纠正批次补入甲另一用户第 2 天有效记录，并送一条把乙用户错归甲的拒绝记录；
- 甲第 3 天留存由 1/2 变 2/2，乙保持 0/2；
- 复盘从同一批次读取两轮留存，逐期差值、实际新增回访人数与排名正确；
- 群体和观察期不被外群更正改写，并发重送同一回访只计一次；
- 继续引用旧版批次时仍显示当时的合法分母。
"""
from __future__ import annotations

import pytest

from app.retention_correction import (
    RetentionCorrectionEngine,
    RevisitRecord,
)
from app.services.retention_correction import RetentionCorrectionService
from app.store import store


WINDOWS = [1, 2, 3, 7]


@pytest.fixture
def engine():
    return RetentionCorrectionEngine()


@pytest.fixture
def populated_engine(engine):
    engine.register_cohort("COH_A", "甲群", ["a1", "a2"], WINDOWS)
    engine.register_cohort("COH_B", "乙群", ["b1", "b2"], WINDOWS)
    # 甲有一人（a1）第 3 天回访；乙无人回访。
    engine.submit_revisit(RevisitRecord("r-a1-d3", "a1", "COH_A", 3, source="baseline"))
    return engine


def _rates(round_):
    return {stat.window_day: stat for stat in round_.windows}


class TestBaselineRetention:
    def test_baseline_retention_query(self, populated_engine):
        round_a = populated_engine.query_retention("COH_A")
        stats_a = _rates(round_a)
        # 第 1、2 天无人回访；第 3 天 1/2
        assert stats_a[1].active_count == 0
        assert stats_a[2].active_count == 0
        assert stats_a[3].active_count == 1
        assert stats_a[3].retention_rate == 0.5
        assert round_a.member_base_count == 2

        round_b = populated_engine.query_retention("COH_B")
        assert all(stat.active_count == 0 for stat in round_b.windows)
        assert all(stat.member_base_count == 2 for stat in round_b.windows)

    def test_observation_windows_frozen_on_register(self, engine):
        engine.register_cohort("COH_X", "X群", ["x1"], [3, 1, 3, 7])
        profile = engine.cohorts["COH_X"]
        assert profile.observation_windows == (1, 3, 7)
        with pytest.raises(ValueError):
            engine.register_cohort("COH_Y", "Y群", ["y1"], [])


class TestCorrectionBatch:
    def test_batch_accepts_valid_and_rejects_cross_cohort(self, populated_engine):
        batch = populated_engine.submit_correction_batch([
            # 补入甲另一用户 a2 第 2 天的有效记录
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2, source="late_feed"),
            # 错把乙用户 b1 归到甲：必须拒绝并留痕
            RevisitRecord("r-b1-wrong", "b1", "COH_A", 2, source="misroute"),
        ])

        assert len(batch.accepted) == 1
        accepted = batch.accepted[0]
        assert accepted.record_id == "r-a2-d2"
        # 逐条关联到对应群的观察期：第 2 天回访影响第 2、3、7 期
        assert accepted.cohort_id == "COH_A"
        assert accepted.affected_windows == (2, 3, 7)

        assert len(batch.rejected) == 1
        rejected = batch.rejected[0]
        assert rejected.record_id == "r-b1-wrong"
        assert rejected.claimed_cohort_id == "COH_A"
        assert rejected.actual_cohort_id == "COH_B"
        assert "外群" in rejected.reason

    def test_cohort_a_day3_changes_from_half_to_full(self, populated_engine):
        before_a = populated_engine.query_retention("COH_A")
        assert _rates(before_a)[3].retention_rate == 0.5

        populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
            RevisitRecord("r-b1-wrong", "b1", "COH_A", 2),
        ])

        after_a = populated_engine.query_retention("COH_A")
        stats_a = _rates(after_a)
        # 第 2 天从 0/2 变 1/2，第 3 天由 1/2 变 2/2，分母始终为 2
        assert stats_a[2].active_count == 1
        assert stats_a[2].retention_rate == 0.5
        assert stats_a[3].active_count == 2
        assert stats_a[3].retention_rate == 1.0
        assert all(stat.member_base_count == 2 for stat in after_a.windows)

    def test_cohort_b_unchanged_after_cross_cohort_rejection(self, populated_engine):
        populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
            RevisitRecord("r-b1-wrong", "b1", "COH_A", 2),
        ])
        round_b = populated_engine.query_retention("COH_B")
        # 乙的观察期与成员不被外群更正改写
        assert tuple(stat.window_day for stat in round_b.windows) == tuple(WINDOWS)
        assert round_b.member_base_count == 2
        assert all(stat.active_count == 0 for stat in round_b.windows)

    def test_concurrent_resubmit_counts_once(self, populated_engine):
        first = populated_engine.submit_revisit(
            RevisitRecord("dup-1", "a2", "COH_A", 2)
        )
        # 并发重送：不同 record_id 但同一回访自然键（群+用户+天数）
        second = populated_engine.submit_revisit(
            RevisitRecord("dup-2", "a2", "COH_A", 2)
        )
        third = populated_engine.submit_revisit(
            RevisitRecord("dup-1", "a2", "COH_A", 2)
        )
        assert first[0] == "accepted"
        assert second[0] == "duplicate"
        assert third[0] == "duplicate"

        batch = populated_engine.submit_correction_batch([
            RevisitRecord("dup-3", "a2", "COH_A", 2),
            RevisitRecord("dup-4", "a2", "COH_A", 2),
        ])
        assert batch.duplicate_count == 2
        assert batch.accepted == []
        round_a = populated_engine.query_retention("COH_A")
        # 第 2 天回访人数没有因为重送而增加
        assert _rates(round_a)[2].active_count == 1
        assert _rates(round_a)[3].active_count == 2

    def test_rejected_records_do_not_mutate_other_cohort(self, populated_engine):
        batch = populated_engine.submit_correction_batch([
            RevisitRecord("r-b2-wrong", "b2", "COH_A", 1),
            RevisitRecord("r-a1-wrong-b", "a1", "COH_B", 1),
        ])
        assert len(batch.rejected) == 2
        # 被拒记录同时波及两个群的复盘范围，但不留任何有效回访
        assert set(batch.involved_cohort_ids) == {"COH_A", "COH_B"}
        stats_a = _rates(populated_engine.query_retention("COH_A"))
        stats_b = _rates(populated_engine.query_retention("COH_B"))
        assert stats_a[3].active_count == 1
        assert all(stat.active_count == 0 for stat in stats_b.values())


class TestReview:
    def test_review_reads_two_rounds_and_computes_deltas(self, populated_engine):
        batch = populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
            RevisitRecord("r-b1-wrong", "b1", "COH_A", 2),
        ])
        review = populated_engine.run_review(batch.batch_id)

        cohort_by_id = {item["cohort_id"]: item for item in review["cohorts"]}
        entry_a = cohort_by_id["COH_A"]
        window_a = {item["window_day"]: item for item in entry_a["windows"]}

        # 第 2 天：0/2 -> 1/2，差值 +0.5；第 3 天：1/2 -> 2/2，差值 +0.5
        assert window_a[2]["before_active_count"] == 0
        assert window_a[2]["after_active_count"] == 1
        assert window_a[2]["active_delta"] == 1
        assert window_a[2]["retention_delta"] == 0.5
        assert window_a[3]["before_active_count"] == 1
        assert window_a[3]["after_active_count"] == 2
        assert window_a[3]["retention_delta"] == 0.5
        # 只保留实际影响的观察期
        assert entry_a["affected_windows"] == [2, 3, 7]
        assert entry_a["newly_added_revisit_users"] == 1

        entry_b = cohort_by_id["COH_B"]
        assert entry_b["newly_added_revisit_users"] == 0
        assert all(item["active_delta"] == 0 for item in entry_b["windows"])

    def test_review_ranking_lists_only_changed_cohorts(self, populated_engine):
        batch = populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
            RevisitRecord("r-b1-wrong", "b1", "COH_A", 2),
        ])
        review = populated_engine.run_review(batch.batch_id)

        # 总排名只列实际新增回访的群：只有甲
        assert review["ranked_cohort_ids"] == ["COH_A"]
        assert review["overall_ranking"][0]["newly_added_revisit_users"] == 1
        # 各期排名同样只列发生差值的群
        for day in (2, 3, 7):
            rows = review["window_rankings"][str(day)]
            assert [row["cohort_id"] for row in rows] == ["COH_A"]
        # 第 1 天没有变化，不应出现在排名里
        assert "1" not in review["window_rankings"]

    def test_review_keeps_rejected_records(self, populated_engine):
        batch = populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
            RevisitRecord("r-b1-wrong", "b1", "COH_A", 2),
        ])
        review = populated_engine.run_review(batch.batch_id)
        assert len(review["rejected_records"]) == 1
        rejected = review["rejected_records"][0]
        assert rejected["record_id"] == "r-b1-wrong"
        assert rejected["actual_cohort_id"] == "COH_B"
        assert review["duplicate_count"] == 0

    def test_verification_status_and_publish_flow(self, populated_engine):
        batch = populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
            RevisitRecord("r-b1-wrong", "b1", "COH_A", 2),
        ])
        # 甲的更正已核实；乙的回访来源部分失效且仍待补证
        populated_engine.mark_pending_evidence(batch.batch_id, ["COH_B"])
        review = populated_engine.run_review(batch.batch_id)
        assert review["review_status"] == "待发布"

        statuses = {item["cohort_id"]: item["verification_status"] for item in review["cohorts"]}
        assert statuses["COH_A"] == "已核实"
        assert statuses["COH_B"] == "待补证"

        # 值班员打开比较结果但尚未发布
        assert populated_engine.get_batch(batch.batch_id).review_status == "待发布"
        published = populated_engine.publish_review(batch.batch_id)
        assert published["review_status"] == "已发布"
        assert populated_engine.get_batch(batch.batch_id).status == "已发布"

    def test_review_is_deterministic(self, populated_engine):
        batch = populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
        ])
        r1 = populated_engine.run_review(batch.batch_id)
        r2 = populated_engine.run_review(batch.batch_id)
        assert r1 == r2


class TestHistoricalDenominator:
    def test_old_batch_keeps_legal_denominator(self, populated_engine):
        first_batch = populated_engine.submit_correction_batch([
            RevisitRecord("r-a2-d2", "a2", "COH_A", 2),
        ])
        history_after_first = populated_engine.cohort_history("COH_A")
        assert len(history_after_first) == 1
        first_snapshot = history_after_first[0]
        # 旧版批次里的合法分母始终是群成员数 2
        assert first_snapshot["member_base_count"] == 2
        assert all(item["member_base_count"] == 2 for item in first_snapshot["after_windows"])

        # 第二个批次再补数据，引用旧版批次仍显示当时口径
        populated_engine.submit_correction_batch([
            RevisitRecord("r-a1-d1", "a1", "COH_A", 1),
        ])
        history = populated_engine.cohort_history("COH_A")
        assert len(history) == 2
        old = next(item for item in history if item["batch_id"] == first_batch.batch_id)
        old_day3 = next(item for item in old["after_windows"] if item["window_day"] == 3)
        assert old_day3["active_count"] == 2
        assert old_day3["member_base_count"] == 2
        assert old_day3["retention_rate"] == 1.0

    def test_unknown_cohort_and_batch_raise(self, populated_engine):
        with pytest.raises(KeyError):
            populated_engine.query_retention("COH_MISSING")
        with pytest.raises(KeyError):
            populated_engine.run_review("RCB-MISSING")


class TestServiceLayer:
    @pytest.fixture
    def service(self, populated_engine):
        # 服务复用测试引擎，避免全局单例状态串扰；同时清空仓库镜像
        store.rows("retention_correction").clear()
        return RetentionCorrectionService(engine=populated_engine)

    @staticmethod
    def _mirrored_row(batch_id: str):
        return [
            row for row in store.rows("retention_correction")
            if row["批次编号"] == batch_id
        ][-1]

    def test_submit_batch_payload(self, service):
        result = service.submit_batch({
            "records": [
                {"record_id": "r-a2-d2", "user_id": "a2",
                 "claimed_cohort_id": "COH_A", "revisit_day": 2},
                {"record_id": "r-b1-wrong", "user_id": "b1",
                 "claimed_cohort_id": "COH_A", "revisit_day": 2},
            ],
        })
        assert result["ok"] is True
        batch_row = result["batch"]
        assert batch_row["accepted_count"] == 1
        assert batch_row["rejected_count"] == 1

        # 批次已镜像到通用资源仓库，可走统一列表/导出路由
        rows = store.rows("retention_correction")
        mirrored = self._mirrored_row(batch_row["batch_id"])
        assert mirrored["接受回访"] == 1
        assert mirrored["拒绝记录"] == 1
        assert mirrored["abnormal"] is True

    def test_service_review_and_publish_actions(self, service):
        created = service.submit_batch({
            "records": [
                {"user_id": "a2", "claimed_cohort_id": "COH_A", "revisit_day": 2},
            ],
            "pending_evidence_cohorts": ["COH_A"],
        })
        batch_id = created["batch"]["batch_id"]
        review = service.review_batch(batch_id)
        assert review["review_status"] == "待发布"

        row = self._mirrored_row(batch_id)
        entry, message = service.run_action(row["id"], "发布复盘")
        # review 已由 review_batch 生成，可直接发布
        assert entry is not None
        assert "发布" in message
        assert entry["复盘状态"] == "已发布"
        assert entry["status"] == "已发布"

    def test_publish_without_review_rejected(self, service):
        created = service.submit_batch({
            "records": [
                {"user_id": "a2", "claimed_cohort_id": "COH_A", "revisit_day": 2},
            ],
        })
        batch_id = created["batch"]["batch_id"]
        # 直接通过动作发布一个未复盘的新批次：先制造一个未复盘批次
        fresh = service.engine.submit_correction_batch([
            RevisitRecord("r-a1-d1", "a1", "COH_A", 1),
        ])
        service._mirror_batch(fresh)
        row = self._mirrored_row(fresh.batch_id)
        entry, message = service.run_action(row["id"], "发布复盘")
        assert entry is None
        assert "先执行复盘" in message

    def test_invalid_record_payload(self, service):
        result = service.submit_batch({"records": [{"user_id": "a2"}]})
        assert result["ok"] is False

    def test_retention_query_endpoint_shape(self, service):
        payload = service.query_retention("COH_A")
        assert payload["cohort_id"] == "COH_A"
        assert payload["member_base_count"] == 2
        assert [item["window_day"] for item in payload["windows"]] == WINDOWS
