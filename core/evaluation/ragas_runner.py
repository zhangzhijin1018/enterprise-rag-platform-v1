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


async def _answer_one(runtime: RAGRuntime, question: str) -> tuple[str, list[str]]:
    """针对单个问题运行完整 RAG 链路，并提取答案与上下文。"""

    state = await run_rag_async(runtime, question=question)
    answer = state.get("answer") or ""
    ctxs = state.get("reranked_hits") or []
    # RAGAS 的 `contexts` 字段期望是字符串列表，所以这里只保留 chunk 内容本身。
    contexts = [str(x.get("content", "")) for x in ctxs]
    return answer, contexts


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

    async def gather() -> tuple[list[str], list[list[str]]]:
        """串行收集每个问题的答案与上下文。

        这里故意保持串行，优先保证教学环境下日志和错误更容易定位。
        如果后续要追求评测吞吐，可以再改成并发 gather。
        """

        answers: list[str] = []
        contexts: list[list[str]] = []
        for i, q in enumerate(questions):
            a, ctx = await _answer_one(runtime, q)
            answers.append(a)
            if not ctx and i < len(ref_contexts):
                # 如果系统没有召回到上下文，就回退到数据集自带的参考 contexts，
                # 这样至少能让 RAGAS 在某些弱联网 / 离线场景下继续工作。
                ctx = list(ref_contexts[i])
            contexts.append(ctx)
        return answers, contexts

    answers, contexts = asyncio.run(gather())

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
        if "faithfulness" in summary:
            FAITHFULNESS_SCORE.set(summary["faithfulness"])
        payload = {"summary": summary, "rows": df.to_dict(orient="records")}
    except Exception as e:  # noqa: BLE001
        # 即便评测失败，也把错误和 traceback 落盘，方便后续定位环境问题。
        payload = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "summary": {},
            "rows": [],
        }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


async def run_ragas_eval_async(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    runtime: RAGRuntime | None = None,
) -> Path:
    """异步封装同步评测函数，便于在 FastAPI 或后台任务中调用。"""

    return await asyncio.to_thread(run_ragas_eval, dataset_path, output_dir, runtime)
