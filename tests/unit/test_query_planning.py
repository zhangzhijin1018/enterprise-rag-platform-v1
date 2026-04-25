"""查询规划与澄清判定单元测试。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock
import json as _json

import pytest

from core.config.settings import get_settings
from core.generation.llm_client import LLMClient
from core.orchestration.nodes.analyze_query import analyze_query_node
from core.orchestration.query_understanding_vocab import load_query_understanding_vocab
from core.orchestration.nodes.clarify_query import clarify_query_node
from core.orchestration.nodes.resolve_context import resolve_context_node
from core.orchestration.query_expansion import build_query_plan
from core.retrieval.cache import RedisCache


class _FakeLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.enabled = True

    async def complete(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return json.dumps(self.payload, ensure_ascii=False), {"mock": True}


def _runtime() -> MagicMock:
    runtime = MagicMock()
    runtime.settings = get_settings()
    runtime.llm = LLMClient(runtime.settings)
    runtime.cache = RedisCache()
    return runtime


@pytest.mark.asyncio
async def test_clarify_node_flags_missing_slots_for_schedule_question() -> None:
    runtime = _runtime()
    state = {
        "question": "今天谁值班？",
        "strategy_signals": {"need_history_resolution": True},
    }
    out = await clarify_query_node(state, runtime)
    assert out.get("need_clarify") is True
    assert "department" in (out.get("missing_slots") or [])
    assert "班次" in (out.get("clarify_question") or "")


@pytest.mark.asyncio
async def test_query_plan_decomposes_compare_question_without_llm() -> None:
    runtime = _runtime()
    plan = await build_query_plan(
        runtime,
        question="Milvus 和 Zilliz Cloud 有什么区别？",
        strategy_signals={
            "need_sub_queries": True,
            "need_keyword_boost": True,
            "need_hyde": False,
            "likely_structured_lookup": False,
            "has_precise_identifier": True,
        },
    )
    assert "区别" in plan.rewritten_query
    assert len(plan.multi_queries) >= 2
    assert any("Milvus" in q for q in plan.keyword_queries)


@pytest.mark.asyncio
async def test_query_plan_extracts_structured_filters_for_fact_lookup() -> None:
    runtime = _runtime()
    plan = await build_query_plan(
        runtime,
        question="今天冲压二车间谁上夜班",
        strategy_signals={
            "need_sub_queries": False,
            "need_keyword_boost": True,
            "need_hyde": False,
            "likely_structured_lookup": True,
            "has_precise_identifier": True,
        },
    )
    assert plan.structured_filters.get("time") == "今天"
    assert plan.structured_filters.get("department") == "冲压二车间"
    assert plan.structured_filters.get("shift") == "夜班"
    assert any("冲压二车间" in q for q in plan.keyword_queries)


@pytest.mark.asyncio
async def test_resolve_context_node_builds_resolved_query_from_history() -> None:
    runtime = _runtime()
    state = {
        "question": "今天是谁值班",
        "strategy_signals": {"need_history_resolution": True},
        "history_messages": [
            {"role": "user", "content": "冲压二车间夜班安排表在哪"},
            {"role": "assistant", "content": "请看共享盘排班目录"},
        ],
    }
    out = await resolve_context_node(state, runtime)
    resolved_query = out.get("resolved_query") or ""
    assert "冲压二车间" in resolved_query
    assert "夜班" in resolved_query


@pytest.mark.asyncio
async def test_analyze_query_node_extracts_enterprise_metadata_intent_and_profiles() -> None:
    out = await analyze_query_node({"question": "设备巡检 SOP 的锅炉巡检步骤是什么"})

    signals = out["strategy_signals"]
    assert signals["query_scene"] == "procedure_lookup"
    assert signals["preferred_retriever"] == "hybrid"
    assert signals["top_k_profile"] == "balanced"
    assert out["metadata_intent"]["business_domain"] == "equipment_maintenance"
    assert out["metadata_intent"]["process_stage"] == "inspection"
    assert "锅炉" in out["metadata_intent"]["equipment_type"]
    assert out["analysis_source"] == "heuristic"
    assert out["analysis_confidence"] > 0.7


@pytest.mark.asyncio
async def test_analyze_query_node_uses_llm_enhancement_when_heuristic_confidence_is_low() -> None:
    runtime = _runtime()
    runtime.llm = _FakeLLM(
        {
            "need_history_resolution": True,
            "need_sub_queries": True,
            "need_hyde": False,
            "need_keyword_boost": True,
            "query_scene": "project_trace",
            "preferred_retriever": "hybrid",
            "top_k_profile": "broad",
            "metadata_intent": {"business_domain": "project_management"},
            "confidence": 0.83,
            "reason": "问题包含历史指代和方案变更信号。",
        }
    )

    out = await analyze_query_node(
        {"question": "上次说那个锅炉相关的方案，后来是不是按新的检查要求改过"},
        runtime,
    )

    signals = out["strategy_signals"]
    assert out["analysis_source"] == "llm_enhanced"
    assert out["analysis_confidence"] >= 0.8
    assert signals["query_scene"] == "project_trace"
    assert signals["top_k_profile"] == "broad"
    assert out["metadata_intent"]["business_domain"] == "project_management"


@pytest.mark.asyncio
async def test_analyze_query_node_applies_guardrail_for_very_low_confidence() -> None:
    runtime = _runtime()
    runtime.llm = _FakeLLM(
        {
            "need_history_resolution": False,
            "need_sub_queries": True,
            "need_hyde": True,
            "need_keyword_boost": False,
            "query_scene": "project_trace",
            "preferred_retriever": "dense",
            "top_k_profile": "broad",
            "metadata_intent": {},
            "confidence": 0.42,
            "reason": "问题指代较强，判断不稳定。",
        }
    )

    out = await analyze_query_node({"question": "那个后来是不是改过"}, runtime)

    signals = out["strategy_signals"]
    assert out["analysis_source"].endswith("_guardrail")
    assert signals["preferred_retriever"] == "hybrid"
    assert signals["top_k_profile"] == "balanced"
    assert signals["need_hyde"] is False


def test_load_query_understanding_vocab_merges_custom_file(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "query_vocab.json"
    path.write_text(
        _json.dumps(
            {
                "equipment_keywords": ["脱硫塔"],
                "project_keywords": ["技改专项"],
                "business_domain_keywords": {
                    "environmental_protection": ["脱硫", "脱硝"]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = get_settings().model_copy(update={"query_understanding_vocab_path": str(path)})

    vocab = load_query_understanding_vocab(settings)

    assert "脱硫塔" in vocab["equipment_keywords"]
    assert "技改专项" in vocab["project_keywords"]
    assert "environmental_protection" in vocab["business_domain_keywords"]


@pytest.mark.asyncio
async def test_analyze_query_node_uses_custom_vocab_file_for_domain_terms(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "query_vocab.json"
    path.write_text(
        _json.dumps(
            {
                "equipment_keywords": ["脱硫塔"],
                "business_domain_keywords": {
                    "environmental_protection": ["脱硫", "脱硝"]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime = _runtime()
    runtime.settings = runtime.settings.model_copy(
        update={"query_understanding_vocab_path": str(path)}
    )

    out = await analyze_query_node({"question": "脱硫塔检修步骤是什么"}, runtime)

    assert out["metadata_intent"]["equipment_type"] == "脱硫塔"
    assert out["metadata_intent"]["business_domain"] == "environmental_protection"


@pytest.mark.asyncio
async def test_analyze_query_node_uses_enterprise_aliases_for_department_site_and_system(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "query_vocab.json"
    path.write_text(
        _json.dumps(
            {
                "department_aliases": {
                    "安全环保部": ["安环部"]
                },
                "site_aliases": {
                    "准东二矿": ["二矿"]
                },
                "system_aliases": {
                    "安全生产管理平台": ["安生平台"]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime = _runtime()
    runtime.settings = runtime.settings.model_copy(
        update={"query_understanding_vocab_path": str(path)}
    )

    out = await analyze_query_node({"question": "安环部在二矿用安生平台看隐患排查记录怎么查"}, runtime)

    intent = out["metadata_intent"]
    assert intent["department"] == "安全环保部"
    assert intent["owner_department"] == "安全环保部"
    assert intent["plant"] == "准东二矿"
    assert intent["applicable_site"] == "准东二矿"
    assert intent["system_name"] == "安全生产管理平台"


@pytest.mark.asyncio
async def test_analyze_query_node_uses_default_xinjiang_energy_vocab_for_fuel_management() -> None:
    out = await analyze_query_node({"question": "燃管部在一号输煤线用燃料平台查入厂煤采样记录怎么查"})

    intent = out["metadata_intent"]
    assert intent["department"] == "燃料管理部"
    assert intent["owner_department"] == "燃料管理部"
    assert intent["plant"] == "一号输煤线"
    assert intent["applicable_site"] == "一号输煤线"
    assert intent["system_name"] == "燃料管控平台"
    assert intent["business_domain"] == "fuel_management"
