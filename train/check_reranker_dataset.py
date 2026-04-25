"""检查 reranker 训练集质量的辅助脚本。

这个脚本不负责训练，只负责回答一个很现实的问题：
“我手上的这批 triplet 数据，值不值得直接拿去开第一轮训练？”

它会做 4 类检查：
1. 基础规模：总行数、正负样本数、平均长度；
2. 结构完整性：query / positive / negative(s) 是否齐全；
3. 重复与脏数据：空文本、正负例重复、同 query 反复出现；
4. 元信息覆盖：scenario / tags / source 是否有保留。

输出方式尽量保持简单：
- 终端打印摘要；
- 可选写一个 JSON 报告，便于后续留档或自动化对比。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """读取 JSONL 数据集。"""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    """做轻量文本归一，便于重复检测。"""
    return " ".join((text or "").strip().split())


def normalize_row(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    """统一解析训练样本里的 query / positive / negatives。"""
    query = normalize_text(str(row.get("query", "")))
    positive = normalize_text(str(row.get("positive", "")))

    negatives: list[str] = []
    negative = row.get("negative")
    if isinstance(negative, str):
        text = normalize_text(negative)
        if text:
            negatives.append(text)

    extra_negatives = row.get("negatives")
    if isinstance(extra_negatives, list):
        for item in extra_negatives:
            text = normalize_text(str(item))
            if text:
                negatives.append(text)
    return query, positive, negatives


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构造样本质量报告。

    报告重点不是做学术级数据审计，而是回答：
    - 样本够不够；
    - 脏不脏；
    - 有没有明显不该直接开训的问题。
    """
    invalid_rows: list[int] = []
    duplicate_positive_negative_rows: list[int] = []
    query_counter: Counter[str] = Counter()
    scenario_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    negative_count_distribution: Counter[int] = Counter()

    query_lengths: list[int] = []
    positive_lengths: list[int] = []
    negative_lengths: list[int] = []

    rows_with_scenario = 0
    rows_with_tags = 0
    rows_with_source = 0

    unique_triplets: set[tuple[str, str, tuple[str, ...]]] = set()
    duplicate_triplet_rows: list[int] = []

    for idx, row in enumerate(rows, start=1):
        query, positive, negatives = normalize_row(row)
        if not query or not positive or not negatives:
            invalid_rows.append(idx)
            continue

        query_counter[query] += 1
        query_lengths.append(len(query))
        positive_lengths.append(len(positive))
        negative_count_distribution[len(negatives)] += 1

        for negative in negatives:
            negative_lengths.append(len(negative))
            if negative == positive:
                duplicate_positive_negative_rows.append(idx)
                break

        scenario = normalize_text(str(row.get("scenario", "")))
        if scenario:
            rows_with_scenario += 1
            scenario_counter[scenario] += 1

        tags = row.get("tags") or []
        if isinstance(tags, list) and tags:
            rows_with_tags += 1
            for tag in tags:
                text = normalize_text(str(tag))
                if text:
                    tag_counter[text] += 1

        source = normalize_text(str(row.get("source", "")))
        if source:
            rows_with_source += 1
            source_counter[source] += 1

        triplet_key = (query, positive, tuple(sorted(set(negatives))))
        if triplet_key in unique_triplets:
            duplicate_triplet_rows.append(idx)
        else:
            unique_triplets.add(triplet_key)

    report = {
        "total_rows": len(rows),
        "valid_rows": len(rows) - len(invalid_rows),
        "invalid_rows": invalid_rows,
        "duplicate_positive_negative_rows": duplicate_positive_negative_rows,
        "duplicate_triplet_rows": duplicate_triplet_rows,
        "unique_queries": len(query_counter),
        "top_queries": query_counter.most_common(10),
        "avg_query_chars": round(mean(query_lengths), 2) if query_lengths else 0,
        "avg_positive_chars": round(mean(positive_lengths), 2) if positive_lengths else 0,
        "avg_negative_chars": round(mean(negative_lengths), 2) if negative_lengths else 0,
        "negative_count_distribution": dict(negative_count_distribution),
        "rows_with_scenario": rows_with_scenario,
        "rows_with_tags": rows_with_tags,
        "rows_with_source": rows_with_source,
        "top_scenarios": scenario_counter.most_common(10),
        "top_sources": source_counter.most_common(10),
        "top_tags": tag_counter.most_common(20),
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    """把核心指标打印到终端。

    这里刻意不打印太多细节，避免人一眼看不清重点。
    更完整的数据仍然会保留在 JSON 输出里。
    """
    print("=== reranker dataset quality report ===")
    print(f"总行数: {report['total_rows']}")
    print(f"有效行数: {report['valid_rows']}")
    print(f"无效行数: {len(report['invalid_rows'])}")
    print(f"唯一 query 数: {report['unique_queries']}")
    print(f"平均 query 长度(字符): {report['avg_query_chars']}")
    print(f"平均正例长度(字符): {report['avg_positive_chars']}")
    print(f"平均负例长度(字符): {report['avg_negative_chars']}")
    print(f"负例数量分布: {report['negative_count_distribution']}")
    print(f"带 scenario 的样本数: {report['rows_with_scenario']}")
    print(f"带 tags 的样本数: {report['rows_with_tags']}")
    print(f"带 source 的样本数: {report['rows_with_source']}")
    print(f"正负例重复行数: {len(report['duplicate_positive_negative_rows'])}")
    print(f"重复 triplet 行数: {len(report['duplicate_triplet_rows'])}")
    print(f"Top scenarios: {report['top_scenarios']}")
    print(f"Top tags: {report['top_tags'][:10]}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="检查 reranker 训练集质量，帮助判断这批样本是否适合直接开训。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="待检查的 reranker 训练集 JSONL 路径。",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="可选，把完整检查结果额外写成 JSON 报告。",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主入口。"""
    args = parse_args()
    rows = load_jsonl(args.dataset_path)
    if not rows:
        raise ValueError("训练集为空，无法做质量检查。")

    report = build_report(rows)
    print_report(report)

    if args.report_path:
        out_path = Path(args.report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"完整报告已写入: {out_path}")


if __name__ == "__main__":
    main()
