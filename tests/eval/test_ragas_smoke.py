"""RAGAS 评测链路的烟雾测试。用于验证评测入口能够正常跑通。"""

import os
import pytest


pytest.importorskip("ragas")


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY for RAGAS+LLM"
)
def test_ragas_minimal_runs() -> None:
    from pathlib import Path

    from core.evaluation.ragas_runner import run_ragas_eval
    from core.services.runtime import reset_runtime

    reset_runtime()
    p = (
        Path(__file__).resolve().parents[2]
        / "core/evaluation/datasets/sample_eval.jsonl"
    )
    out = run_ragas_eval(dataset_path=p)
    assert out.exists()
