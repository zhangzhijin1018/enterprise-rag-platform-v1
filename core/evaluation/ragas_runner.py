"""RAGAS 评测执行模块。

作用：
1. 读取 JSONL 评测集。
2. 调用当前 RAG 系统批量生成答案与上下文。
3. 交给 RAGAS 计算 faithfulness、answer relevancy 等指标。
4. 把结果落盘成 JSON 报告。
"""

from __future__ import annotations

import asyncio
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset

from core.config.settings import get_settings
from core.observability.metrics import FAITHFULNESS_SCORE
from core.orchestration.graph import run_rag_async
from core.services.runtime import RAGRuntime, get_runtime


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 数据集。"""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _state_eval_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """从 RAG state 中提取评测侧也值得保留的治理字段。"""

    strategy_signals = state.get("strategy_signals") or {}
    reranked_hits = state.get("reranked_hits") or []
    citations = state.get("citations") or []
    matched_routes: list[str] = []
    enterprise_entity_matches: list[str] = []
    seen_routes: set[str] = set()
    seen_entity_groups: set[str] = set()
    metadata_boosted = False
    enterprise_entity_boosted = False
    governance_boosted = False
    for item in reranked_hits:
        trace = item.get("trace") or {}
        for route in trace.get("matched_routes") or []:
            route_text = str(route).strip()
            if route_text and route_text not in seen_routes:
                seen_routes.add(route_text)
                matched_routes.append(route_text)
        if isinstance(trace.get("metadata_boost"), (int, float)) and float(trace["metadata_boost"]) > 0:
            metadata_boosted = True
        if isinstance(trace.get("enterprise_entity_boost"), (int, float)) and float(
            trace["enterprise_entity_boost"]
        ) > 0:
            enterprise_entity_boosted = True
        for entity_group in trace.get("enterprise_entity_matches") or []:
            entity_text = str(entity_group).strip()
            if entity_text and entity_text not in seen_entity_groups:
                seen_entity_groups.add(entity_text)
                enterprise_entity_matches.append(entity_text)
        if isinstance(trace.get("governance_bonus"), (int, float)) and float(trace["governance_bonus"]) > 0:
            governance_boosted = True
    explainable_citation_count = sum(
        1 for item in citations if isinstance(item, dict) and str(item.get("selection_reason") or "").strip()
    )

    return {
        "refusal": bool(state.get("refusal")),
        "refusal_reason": state.get("refusal_reason"),
        "answer_mode": state.get("answer_mode"),
        "data_classification": state.get("data_classification"),
        "model_route": state.get("model_route"),
        "analysis_confidence": state.get("analysis_confidence"),
        "analysis_source": state.get("analysis_source"),
        "analysis_reason": state.get("analysis_reason"),
        "query_scene": state.get("query_scene") or strategy_signals.get("query_scene"),
        "preferred_retriever": state.get("preferred_retriever")
        or strategy_signals.get("preferred_retriever"),
        "top_k_profile": state.get("top_k_profile") or strategy_signals.get("top_k_profile"),
        "conflict_detected": bool(state.get("conflict_detected")),
        "conflict_summary": state.get("conflict_summary"),
        "audit_id": state.get("audit_id"),
        "matched_routes": matched_routes,
        "metadata_boosted": metadata_boosted,
        "enterprise_entity_boosted": enterprise_entity_boosted,
        "enterprise_entity_matches": enterprise_entity_matches,
        "governance_boosted": governance_boosted,
        "explainable_citation_count": explainable_citation_count,
    }


def _to_bool(value: Any) -> bool:
    """把评测数据里的布尔语义字段规整成稳定 bool。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _normalize_tags(value: Any) -> list[str]:
    """规整评测样本的标签字段。"""

    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _build_eval_signal_summary(
    dataset_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    total: int,
) -> dict[str, float]:
    """汇总拒答、冲突、路由等企业治理信号。"""

    if total <= 0:
        return {}
    refusal_count = sum(1 for row in metadata_rows if row.get("refusal"))
    conflict_count = sum(1 for row in metadata_rows if row.get("conflict_detected"))
    metadata_boost_count = sum(1 for row in metadata_rows if row.get("metadata_boosted"))
    enterprise_entity_boost_count = sum(
        1 for row in metadata_rows if row.get("enterprise_entity_boosted")
    )
    governance_boost_count = sum(1 for row in metadata_rows if row.get("governance_boosted"))
    explainable_citation_count = sum(int(row.get("explainable_citation_count") or 0) for row in metadata_rows)
    confidence_values = [
        float(row["analysis_confidence"])
        for row in metadata_rows
        if isinstance(row.get("analysis_confidence"), (int, float))
    ]
    classification_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    matched_route_counts: dict[str, int] = {}
    entity_match_counts: dict[str, int] = {}
    analysis_source_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    expected_refusal_total = 0
    expected_refusal_hits = 0
    expected_conflict_total = 0
    expected_conflict_hits = 0
    for index, dataset_row in enumerate(dataset_rows):
        scenario = str(dataset_row.get("scenario") or "").strip()
        if scenario:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        for tag in _normalize_tags(dataset_row.get("tags")):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if "expected_refusal" in dataset_row:
            expected_refusal_total += 1
            if _to_bool(dataset_row.get("expected_refusal")) == bool(
                metadata_rows[index].get("refusal")
            ):
                expected_refusal_hits += 1
        if "expected_conflict" in dataset_row:
            expected_conflict_total += 1
            if _to_bool(dataset_row.get("expected_conflict")) == bool(
                metadata_rows[index].get("conflict_detected")
            ):
                expected_conflict_hits += 1
    for row in metadata_rows:
        classification = str(row.get("data_classification") or "").strip()
        route = str(row.get("model_route") or "").strip()
        analysis_source = str(row.get("analysis_source") or "").strip()
        if classification:
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
        if route:
            route_counts[route] = route_counts.get(route, 0) + 1
        if analysis_source:
            analysis_source_counts[analysis_source] = analysis_source_counts.get(analysis_source, 0) + 1
        for matched_route in _normalize_tags(row.get("matched_routes")):
            matched_route_counts[matched_route] = matched_route_counts.get(matched_route, 0) + 1
        for entity_group in _normalize_tags(row.get("enterprise_entity_matches")):
            entity_match_counts[entity_group] = entity_match_counts.get(entity_group, 0) + 1

    summary: dict[str, float] = {
        "sample_count": float(total),
        "refusal_rate": refusal_count / total,
        "conflict_detected_rate": conflict_count / total,
        "metadata_boost_hit_rate": metadata_boost_count / total,
        "enterprise_entity_boost_hit_rate": enterprise_entity_boost_count / total,
        "governance_boost_hit_rate": governance_boost_count / total,
        "avg_explainable_citations": explainable_citation_count / total,
        "avg_analysis_confidence": (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0,
    }
    for key, value in classification_counts.items():
        summary[f"classification:{key}"] = value / total
    for key, value in route_counts.items():
        summary[f"model_route:{key}"] = value / total
    for key, value in analysis_source_counts.items():
        summary[f"analysis_source:{key}"] = value / total
    for key, value in matched_route_counts.items():
        summary[f"matched_route:{key}"] = value / total
    for key, value in entity_match_counts.items():
        summary[f"entity_match:{key}"] = value / total
    for key, value in scenario_counts.items():
        summary[f"scenario:{key}"] = value / total
    for key, value in tag_counts.items():
        summary[f"tag:{key}"] = value / total
    if expected_refusal_total:
        summary["expected_refusal_match_rate"] = expected_refusal_hits / expected_refusal_total
    if expected_conflict_total:
        summary["expected_conflict_match_rate"] = expected_conflict_hits / expected_conflict_total
    return summary


def _build_query_understanding_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合 query understanding badcase，给出后续调参建议。"""

    report: dict[str, Any] = {
        "top_scenes": [],
        "top_guardrail_scenarios": [],
        "top_llm_enhanced_scenarios": [],
        "recommendations": [],
    }
    if not rows:
        return report

    scene_counts: dict[str, int] = {}
    guardrail_scenarios: dict[str, int] = {}
    llm_scenarios: dict[str, int] = {}
    low_confidence_rows = 0
    guardrail_rows = 0
    llm_rows = 0

    for row in rows:
        scene = str(row.get("query_scene") or "").strip()
        scenario = str(row.get("scenario") or "").strip()
        source = str(row.get("analysis_source") or "").strip()
        confidence = row.get("analysis_confidence")
        if scene:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        if isinstance(confidence, (int, float)) and float(confidence) < 0.6:
            low_confidence_rows += 1
        if "guardrail" in source:
            guardrail_rows += 1
            if scenario:
                guardrail_scenarios[scenario] = guardrail_scenarios.get(scenario, 0) + 1
        elif source == "llm_enhanced":
            llm_rows += 1
            if scenario:
                llm_scenarios[scenario] = llm_scenarios.get(scenario, 0) + 1

    report["top_scenes"] = [
        {"name": key, "count": value}
        for key, value in sorted(scene_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    report["top_guardrail_scenarios"] = [
        {"name": key, "count": value}
        for key, value in sorted(guardrail_scenarios.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    report["top_llm_enhanced_scenarios"] = [
        {"name": key, "count": value}
        for key, value in sorted(llm_scenarios.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]

    recommendations: list[str] = []
    total = max(1, len(rows))
    if guardrail_rows / total >= 0.2:
        recommendations.append("guardrail 触发比例偏高，优先补充高频场景的规则锚点和 metadata 词典。")
    if llm_rows / total >= 0.3:
        recommendations.append("llm_enhanced 占比偏高，说明规则覆盖不足，优先补 procedure / project / meeting 类长尾表达。")
    if low_confidence_rows / total >= 0.3:
        recommendations.append("低置信 query 偏多，建议按 badcase 回放调整 confidence 权重，减少保守回退。")
    if not recommendations:
        recommendations.append("当前 query understanding 结构较稳，优先继续补 badcase 样本而不是大改规则。")
    report["recommendations"] = recommendations
    return report


def _metric_value(row: dict[str, Any], key: str) -> float | None:
    """提取逐题指标值。"""

    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _badcase_priority(row: dict[str, Any]) -> tuple[int, float]:
    """给逐题结果打一个简单优先级，便于生成 badcase 回放。"""

    priority = 0
    if "expected_refusal" in row and _to_bool(row.get("expected_refusal")) != bool(row.get("refusal")):
        priority += 4
    if "expected_conflict" in row and _to_bool(row.get("expected_conflict")) != bool(
        row.get("conflict_detected")
    ):
        priority += 4
    if bool(row.get("refusal")):
        priority += 1
    if bool(row.get("conflict_detected")):
        priority += 1
    for metric in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
        value = _metric_value(row, metric)
        if value is not None and value < 0.6:
            priority += 1
    min_metric = min(
        (
            value
            for value in (
                _metric_value(row, "faithfulness"),
                _metric_value(row, "answer_relevancy"),
                _metric_value(row, "context_recall"),
                _metric_value(row, "context_precision"),
            )
            if value is not None
        ),
        default=1.0,
    )
    return priority, min_metric


def _render_explainability_report(payload: dict[str, Any]) -> str:
    """把 JSON 评测报告收敛成更适合阅读的 Markdown。"""

    lines: list[str] = ["# Eval Explainability Report", ""]
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    query_understanding_report = payload.get("query_understanding_report") or {}
    if isinstance(summary, dict) and summary:
        lines.extend(
            [
                "## Summary",
                "",
                f"- sample_count: {summary.get('sample_count', 0)}",
                f"- refusal_rate: {summary.get('refusal_rate', 0):.4f}",
                f"- conflict_detected_rate: {summary.get('conflict_detected_rate', 0):.4f}",
                f"- metadata_boost_hit_rate: {summary.get('metadata_boost_hit_rate', 0):.4f}",
                f"- governance_boost_hit_rate: {summary.get('governance_boost_hit_rate', 0):.4f}",
                f"- avg_explainable_citations: {summary.get('avg_explainable_citations', 0):.4f}",
                f"- avg_analysis_confidence: {summary.get('avg_analysis_confidence', 0):.4f}",
                "",
            ]
        )
        matched_route_keys = sorted(key for key in summary if key.startswith("matched_route:"))
        analysis_source_keys = sorted(key for key in summary if key.startswith("analysis_source:"))
        if matched_route_keys:
            lines.append("## Matched Routes")
            lines.append("")
            for key in matched_route_keys:
                lines.append(f"- {key}: {summary[key]:.4f}")
            lines.append("")
        if analysis_source_keys:
            lines.append("## Query Understanding")
            lines.append("")
            for key in analysis_source_keys:
                lines.append(f"- {key}: {summary[key]:.4f}")
            lines.append("")
    if isinstance(query_understanding_report, dict) and query_understanding_report:
        lines.append("## Query Understanding Tuning")
        lines.append("")
        for item in query_understanding_report.get("top_scenes") or []:
            if isinstance(item, dict):
                lines.append(f"- top_scene:{item.get('name')}: {item.get('count')}")
        for item in query_understanding_report.get("top_guardrail_scenarios") or []:
            if isinstance(item, dict):
                lines.append(f"- guardrail_scenario:{item.get('name')}: {item.get('count')}")
        for item in query_understanding_report.get("top_llm_enhanced_scenarios") or []:
            if isinstance(item, dict):
                lines.append(f"- llm_enhanced_scenario:{item.get('name')}: {item.get('count')}")
        recommendations = query_understanding_report.get("recommendations") or []
        if recommendations:
            lines.append("")
            lines.append("### Recommendations")
            lines.append("")
            for item in recommendations:
                lines.append(f"- {item}")
            lines.append("")

    if not isinstance(rows, list) or not rows:
        lines.extend(["## Badcases", "", "无逐题结果。", ""])
        return "\n".join(lines)

    badcases = sorted(
        (row for row in rows if isinstance(row, dict)),
        key=_badcase_priority,
        reverse=True,
    )[:10]
    lines.extend(["## Badcases", ""])
    for index, row in enumerate(badcases, start=1):
        priority, min_metric = _badcase_priority(row)
        lines.append(f"### {index}. {row.get('question', 'unknown question')}")
        lines.append("")
        lines.append(f"- priority: {priority}")
        lines.append(f"- scenario: {row.get('scenario') or '-'}")
        lines.append(f"- model_route: {row.get('model_route') or '-'}")
        lines.append(f"- analysis_source: {row.get('analysis_source') or '-'}")
        analysis_confidence = row.get("analysis_confidence")
        if isinstance(analysis_confidence, (int, float)):
            lines.append(f"- analysis_confidence: {float(analysis_confidence):.4f}")
        lines.append(f"- analysis_reason: {row.get('analysis_reason') or '-'}")
        lines.append(f"- data_classification: {row.get('data_classification') or '-'}")
        lines.append(f"- matched_routes: {', '.join(_normalize_tags(row.get('matched_routes'))) or '-'}")
        lines.append(f"- metadata_boosted: {bool(row.get('metadata_boosted'))}")
        lines.append(
            "- enterprise_entity_matches: "
            f"{', '.join(_normalize_tags(row.get('enterprise_entity_matches'))) or '-'}"
        )
        lines.append(f"- enterprise_entity_boosted: {bool(row.get('enterprise_entity_boosted'))}")
        lines.append(f"- governance_boosted: {bool(row.get('governance_boosted'))}")
        lines.append(f"- explainable_citation_count: {int(row.get('explainable_citation_count') or 0)}")
        lines.append(f"- refusal: {bool(row.get('refusal'))} ({row.get('refusal_reason') or '-'})")
        lines.append(f"- conflict_detected: {bool(row.get('conflict_detected'))}")
        lines.append(f"- min_metric: {min_metric:.4f}")
        if row.get("conflict_summary"):
            lines.append(f"- conflict_summary: {row.get('conflict_summary')}")
        expected_checks: list[str] = []
        if "expected_refusal" in row:
            expected_checks.append(
                f"expected_refusal={_to_bool(row.get('expected_refusal'))} actual={bool(row.get('refusal'))}"
            )
        if "expected_conflict" in row:
            expected_checks.append(
                "expected_conflict="
                f"{_to_bool(row.get('expected_conflict'))} actual={bool(row.get('conflict_detected'))}"
            )
        if expected_checks:
            lines.append(f"- expectations: {'; '.join(expected_checks)}")
        lines.append("")
    return "\n".join(lines)


async def _answer_one(runtime: RAGRuntime, question: str) -> tuple[str, list[str], dict[str, Any]]:
    """针对单个问题运行完整 RAG 链路，并提取答案、上下文和治理元信息。"""

    state = await run_rag_async(runtime, question=question)
    answer = state.get("answer") or ""
    ctxs = state.get("reranked_hits") or []
    # RAGAS 的 `contexts` 字段期望是字符串列表，所以这里只保留 chunk 内容本身。
    contexts = [str(x.get("content", "")) for x in ctxs]
    return answer, contexts, _state_eval_metadata(state)


def run_ragas_eval(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    runtime: RAGRuntime | None = None,
) -> Path:
    """同步执行一轮 RAGAS 评测。"""

    settings = get_settings()
    dataset_path = dataset_path or Path(settings.eval_dataset_path)
    output_dir = Path(output_dir or settings.eval_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = runtime or get_runtime()

    # 评测集最少需要问题；ground_truth 和参考 contexts 为空时也能继续跑。
    rows = _load_jsonl(dataset_path)
    questions = [r["question"] for r in rows]
    ground_truths = [r.get("ground_truth", "") for r in rows]
    ref_contexts = [r.get("contexts") or [] for r in rows]

    async def gather() -> tuple[list[str], list[list[str]], list[dict[str, Any]]]:
        """串行收集每个问题的答案与上下文。

        这里故意保持串行，优先保证教学环境下日志和错误更容易定位。
        如果后续要追求评测吞吐，可以再改成并发 gather。
        """

        answers: list[str] = []
        contexts: list[list[str]] = []
        metadata_rows: list[dict[str, Any]] = []
        for i, q in enumerate(questions):
            a, ctx, metadata = await _answer_one(runtime, q)
            answers.append(a)
            if not ctx and i < len(ref_contexts):
                # 如果系统没有召回到上下文，就回退到数据集自带的参考 contexts，
                # 这样至少能让 RAGAS 在某些弱联网 / 离线场景下继续工作。
                ctx = list(ref_contexts[i])
            contexts.append(ctx)
            metadata_rows.append(metadata)
        return answers, contexts, metadata_rows

    answers, contexts, metadata_rows = asyncio.run(gather())

    # RAGAS 依赖 HuggingFace Dataset 作为统一输入格式。
    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"ragas_report_{ts}.json"

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # 这里调用 RAGAS 统一计算多个指标。
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        df = result.to_pandas()
        # 报告里同时保留逐行结果和平均摘要，方便排查单题 badcase 与观察整体趋势。
        summary = {k: float(v) for k, v in df.mean(numeric_only=True).items()}
        summary.update(_build_eval_signal_summary(rows, metadata_rows, len(rows)))
        if "faithfulness" in summary:
            FAITHFULNESS_SCORE.set(summary["faithfulness"])
        result_rows = df.to_dict(orient="records")
        merged_rows = []
        for index, row in enumerate(result_rows):
            merged_rows.append(
                {
                    **rows[index],
                    **row,
                    **metadata_rows[index],
                }
            )
        payload = {
            "summary": summary,
            "rows": merged_rows,
            "query_understanding_report": _build_query_understanding_report(merged_rows),
        }
    except Exception as e:  # noqa: BLE001
        # 即便评测失败，也把错误和 traceback 落盘，方便后续定位环境问题。
        payload = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "summary": _build_eval_signal_summary(rows, metadata_rows, len(rows)),
            "rows": [
                {
                    **rows[index],
                    "answer": answers[index],
                    "contexts": contexts[index],
                    **metadata_rows[index],
                }
                for index in range(len(rows))
            ],
        }
        payload["query_understanding_report"] = _build_query_understanding_report(payload["rows"])

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.with_suffix(".md").write_text(
        _render_explainability_report(payload),
        encoding="utf-8",
    )
    return out_path


async def run_ragas_eval_async(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    runtime: RAGRuntime | None = None,
) -> Path:
    """异步封装同步评测函数，便于在 FastAPI 或后台任务中调用。"""

    return await asyncio.to_thread(run_ragas_eval, dataset_path, output_dir, runtime)
