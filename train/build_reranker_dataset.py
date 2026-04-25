"""从当前项目 `/eval` 报告自动抽取 reranker 训练样本。

这份脚本解决的是一个很现实的问题：
很多团队知道“应该微调 reranker”，但第一步卡在没有训练集。

当前项目已经有两类非常值钱的现成资产：
1. 原始评测集：里面有 question 和相对可信的 contexts；
2. `/eval` 报告：里面有当前系统真实召回到的 contexts 和运行结果。

所以这份脚本采用弱监督思路：
- 正例优先来自评测集 contexts；
- 负例优先来自当前系统召回到、但并非正例的 contexts；
- 先快速构造第一版 triplet，让你能直接开始第一轮 reranker 微调。

这不是最终形态的数据工程，但非常适合当前项目“先把微调链路跑起来，再用 badcase 持续加料”的节奏。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any


def configure_logging() -> None:
    """配置命令行脚本日志。

    这个脚本常见问题通常不是复杂算法错误，而是：
    - 路径不对；
    - 报告格式不对；
    - 样本被过滤光了；
    - 输出文件没有写到预期位置。

    所以简单清晰的 INFO 日志最够用。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_json(path: str) -> Any:
    """读取 JSON 文件。

    这里默认输入的是 `/eval` 产出的 JSON 报告，而不是 JSONL。
    当前项目里的报告结构是：
    - 顶层 `summary`
    - 顶层 `results`
    - 每题一个 result row
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    当前默认读取的是原始评测集 `enterprise_eval.jsonl`。
    每一行通常会包含：
    - question
    - ground_truth
    - contexts
    - scenario
    - tags
    - expected_refusal / expected_conflict
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL 文件。

    输出格式刻意和 `train/train_reranker.py` 兼容，
    这样数据抽样脚本和训练脚本可以直接串起来，中间不用再做格式转换。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """做轻量文本归一，方便去重和对齐。

    这里不做激进清洗，只做：
    - strip
    - 多空白折叠

    原因是：
    - 我们主要想解决“同一段文本因为空白差异而去重失败”的问题；
    - 不想把制度编号、路径箭头、列表结构等对检索有价值的细节误清洗掉。
    """
    return " ".join((text or "").strip().split())


def looks_same_text(left: str, right: str) -> bool:
    """判断两段文本是否可以视为同一内容。

    当前采用严格的归一化后相等判断，而不是相似度判断。

    原因：
    - 第一轮弱监督数据构造优先追求可解释和稳定；
    - 如果这里引入模糊相似度，反而更容易把“其实不同但很像”的文本误判成同一段。
    """
    return normalize_text(left) == normalize_text(right)


def build_reference_map(dataset_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 question 构造评测集参考映射。

    核心假设：
    - `/eval` 报告和原始评测集里的 `question` 能对得上；
    - 对同一个 question，评测集中的 `contexts` 比当前系统实时召回结果更适合作为正例参考。
    """
    mapping: dict[str, dict[str, Any]] = {}
    for row in dataset_rows:
        question = normalize_text(str(row.get("question", "")))
        if not question:
            continue
        mapping[question] = row
    return mapping


def choose_positive_contexts(
    report_row: dict[str, Any],
    reference_row: dict[str, Any] | None,
) -> list[str]:
    """优先使用评测集 `contexts` 作为正例；缺失时退回报告里的 `contexts`。

    为什么优先级这样设计：
    - 原始评测集通常是人工准备或至少经过人工筛过的一组参考上下文；
    - `/eval` 报告里的 `contexts` 是当前系统当时实际召回的内容，其中可能混有错误候选；
    - 所以第一轮弱监督里，正例应尽量站在“参考答案侧”而不是“当前系统侧”。
    """
    positives: list[str] = []
    candidate_sources: list[list[Any]] = []
    if reference_row:
        candidate_sources.append(list(reference_row.get("contexts") or []))
    candidate_sources.append(list(report_row.get("contexts") or []))

    for source in candidate_sources:
        for item in source:
            text = normalize_text(str(item))
            if not text:
                continue
            if any(looks_same_text(text, existed) for existed in positives):
                continue
            positives.append(text)
    return positives


def choose_negative_contexts(
    report_row: dict[str, Any],
    positives: list[str],
    fallback_pool: list[str],
    negatives_per_query: int,
    rng: random.Random,
) -> list[str]:
    """优先从当前题的召回上下文中抽负样本，不够再从跨题上下文补齐。

    负样本优先级为什么这样排：

    1. 当前题召回到的错误候选
       这是最有价值的 hard negative，因为它最接近线上真实误排场景。
    2. 其他题的 contexts
       这是兜底负样本，质量不如 hard negative，但总比没有负样本好。

    参数说明：
    - negatives_per_query:
      每个 query 保留多少个负例。默认 `3` 是一个务实起点，
      足够让训练看到“一个主负例 + 若干补充负例”，又不会让样本体积膨胀太快。
    - rng:
      随机数实例。单独传入而不是直接用全局 random，方便复现和测试。
    """
    negatives: list[str] = []

    for item in list(report_row.get("contexts") or []):
        text = normalize_text(str(item))
        if not text:
            continue
        if any(looks_same_text(text, positive) for positive in positives):
            continue
        if any(looks_same_text(text, existed) for existed in negatives):
            continue
        negatives.append(text)
        if len(negatives) >= negatives_per_query:
            return negatives

    pool = [text for text in fallback_pool if text]
    rng.shuffle(pool)
    for text in pool:
        if any(looks_same_text(text, positive) for positive in positives):
            continue
        if any(looks_same_text(text, existed) for existed in negatives):
            continue
        negatives.append(text)
        if len(negatives) >= negatives_per_query:
            break
    return negatives


def should_skip_row(report_row: dict[str, Any]) -> bool:
    """过滤不适合拿来做 reranker 正负样本的题目。

    当前会跳过两类题：
    1. 明确属于拒答场景的题；
    2. question 本身为空或报告不完整的题。

    这么做的原因是：
    - reranker 训练关注的是“证据排序”；
    - 拒答题很多时候根本没有稳定的正例 chunk，不适合硬塞进当前 triplet 训练；
    - 拒答逻辑更适合在后续单独做拒答分类或生成模型对齐。
    """
    question = normalize_text(str(report_row.get("question", "")))
    if not question:
        return True

    if bool(report_row.get("expected_refusal")):
        return True

    metadata = report_row.get("metadata") or {}
    refusal = bool(report_row.get("refusal"))
    refusal = refusal or bool(metadata.get("refusal"))
    return refusal


def build_triplets(
    report_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    negatives_per_query: int,
    max_rows: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """把 `/eval` 报告和评测集拼成第一版弱监督 triplet 数据。

    输出字段说明：
    - query:
      原始问题文本。
    - positive:
      该问题应命中的参考证据文本。
    - negative:
      主负例。之所以单独保留一个字段，是为了兼容很多常见 triplet 训练格式。
    - negatives:
      其他补充负例。训练脚本会继续把它们展开成多条 pairwise 负样本。
    - scenario:
      问题场景，例如 `policy_lookup`、`procedure_lookup`。
      这个字段当前训练脚本不直接使用，但对后续分桶分析很有用。
    - tags:
      保留原评测集标签，方便后续排查“哪个业务域最容易排错”。
    - source:
      标记样本来源，便于后续混入人工标注样本后做溯源。

    `max_rows` 的意义：
    - 并不是训练必须项；
    - 它主要服务于 smoke test、样本预览和第一轮快速实验。
    """
    rng = random.Random(seed)
    reference_map = build_reference_map(dataset_rows)

    # 跨题上下文池：当某题自己召回到的错误候选太少时，用其他题的 contexts 做弱负例兜底。
    fallback_pool: list[str] = []
    for row in dataset_rows:
        for item in list(row.get("contexts") or []):
            text = normalize_text(str(item))
            if text:
                fallback_pool.append(text)

    triplets: list[dict[str, Any]] = []
    for report_row in report_rows:
        if should_skip_row(report_row):
            continue

        question = normalize_text(str(report_row.get("question", "")))
        reference_row = reference_map.get(question)
        positives = choose_positive_contexts(report_row, reference_row)
        if not positives:
            continue

        negatives = choose_negative_contexts(
            report_row=report_row,
            positives=positives,
            fallback_pool=fallback_pool,
            negatives_per_query=negatives_per_query,
            rng=rng,
        )
        if not negatives:
            continue

        # scenario 不是当前训练损失函数直接使用的字段，但它对后面分析非常重要：
        # 你可以很快统计“policy_lookup 提升了多少”“procedure_lookup 仍然差在哪里”。
        scenario = (
            str(report_row.get("scenario", ""))
            or str((reference_row or {}).get("scenario", ""))
            or str((report_row.get("metadata") or {}).get("query_scene", ""))
        )
        tags = list((reference_row or {}).get("tags") or [])

        for positive in positives:
            if max_rows is not None and len(triplets) >= max_rows:
                return triplets

            triplets.append(
                {
                    "query": question,
                    "positive": positive,
                    "negative": negatives[0],
                    "negatives": negatives[1:],
                    "scenario": scenario,
                    "tags": tags,
                    "source": "eval_report",
                }
            )
    return triplets


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这份脚本的 CLI 设计目标是“先够用，再稳定”：
    - 核心只暴露你最常改的 5 个参数；
    - 尽量和现有项目目录习惯保持一致；
    - 不引入过多策略开关，避免第一版就过度复杂化。
    """
    parser = argparse.ArgumentParser(description="从 `/eval` 报告自动抽取 reranker 训练样本。")
    parser.add_argument(
        "--eval-report-path",
        required=True,
        help="`/eval` 产生的 JSON 报告路径。",
    )
    parser.add_argument(
        "--dataset-path",
        default="./core/evaluation/datasets/enterprise_eval.jsonl",
        help="原始评测集 JSONL 路径，优先用其中 contexts 作为正例。",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="输出的 reranker triplet JSONL 路径。",
    )
    parser.add_argument(
        "--negatives-per-query",
        type=int,
        default=3,
        help="每个 query 最多保留多少个负样本。",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="可选，限制输出的总样本数。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。

    主流程：
    1. 读 `/eval` JSON 报告；
    2. 读原始评测集 JSONL；
    3. 逐题对齐 question；
    4. 构造 positive / negative / negatives；
    5. 输出新的训练集 JSONL。

    这一步的本质，不是做最终精标，而是把你当前已有的评测资产快速转成
    “第一轮可训练数据”，从而降低微调的起步门槛。
    """
    configure_logging()
    args = parse_args()

    logging.info("loading eval report from %s", args.eval_report_path)
    report_payload = load_json(args.eval_report_path)
    report_rows = list(report_payload.get("results") or [])
    if not report_rows:
        raise ValueError("`results` 为空，当前报告不像有效的 `/eval` JSON 输出。")

    logging.info("loading dataset from %s", args.dataset_path)
    dataset_rows = load_jsonl(args.dataset_path)

    # 这里真正发生的是“弱监督标签构造”，而不是简单字段搬运。
    # 也就是说：
    # - 正例来自参考 contexts；
    # - 负例来自真实误召回或跨题兜底；
    # 这样产出的数据比随机凑 query-positive 更接近真实线上 badcase。
    triplets = build_triplets(
        report_rows=report_rows,
        dataset_rows=dataset_rows,
        negatives_per_query=args.negatives_per_query,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    if not triplets:
        raise ValueError("未构造出任何 triplet，请先检查 eval 报告和评测集是否匹配。")

    write_jsonl(args.output_path, triplets)
    logging.info("wrote %s triplets to %s", len(triplets), args.output_path)


if __name__ == "__main__":
    main()
