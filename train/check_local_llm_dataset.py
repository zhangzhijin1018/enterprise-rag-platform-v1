"""检查本地生成模型 SFT 数据质量的辅助脚本。

这个脚本解决的是一个很现实的问题：
即使已经能从 `/eval` 报告自动整理出 `messages` 数据，也不代表这些数据适合直接开训。

它会重点检查：
1. `messages` 结构是否完整；
2. system / user / assistant 三类角色是否齐；
3. grounded / refusal / conflict 样本比例大概怎样；
4. 平均 user / assistant 文本长度是否异常；
5. scenario / tags / source 是否有保留，便于后续分桶分析。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    """做轻量文本归一。"""
    return " ".join((text or "").strip().split())


def extract_role_messages(row: dict[str, Any]) -> tuple[str, str, str]:
    """从单条样本里抽出 system / user / assistant 文本。

    如果样本不合法，这里返回空字符串，后续统一计入 invalid_rows。
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return "", "", ""

    system_parts: list[str] = []
    user_parts: list[str] = []
    assistant_parts: list[str] = []

    for msg in messages:
        role = normalize_text(str(msg.get("role", "")))
        content = normalize_text(str(msg.get("content", "")))
        if not role or not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    return (
        "\n".join(system_parts).strip(),
        "\n".join(user_parts).strip(),
        "\n".join(assistant_parts).strip(),
    )


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构造数据质3量报告。

    报告重点不是做学术级数据治理，而是回答：
    - 当前这批样本能不能直接训练；
    - 如果不太行，最先要清哪类问题。
    """
    invalid_rows: list[int] = []
    duplicate_user_assistant_rows: list[int] = []
    duplicate_sample_rows: list[int] = []

    scenario_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()

    refusal_rows = 0
    conflict_rows = 0
    grounded_rows = 0

    system_lengths: list[int] = []
    user_lengths: list[int] = []
    assistant_lengths: list[int] = []

    unique_keys: set[tuple[str, str, str]] = set()

    for idx, row in enumerate(rows, start=1):
        system_text, user_text, assistant_text = extract_role_messages(row)
        if not user_text or not assistant_text:
            invalid_rows.append(idx)
            continue

        if assistant_text == user_text:
            duplicate_user_assistant_rows.append(idx)

        key = (system_text, user_text, assistant_text)
        if key in unique_keys:
            duplicate_sample_rows.append(idx)
        else:
            unique_keys.add(key)

        system_lengths.append(len(system_text))
        user_lengths.append(len(user_text))
        assistant_lengths.append(len(assistant_text))

        if bool(row.get("refusal")):
            refusal_rows += 1
        elif bool(row.get("conflict_detected")):
            conflict_rows += 1
        else:
            grounded_rows += 1

        scenario = normalize_text(str(row.get("scenario", "")))
        if scenario:
            scenario_counter[scenario] += 1

        source = normalize_text(str(row.get("source", "")))
        if source:
            source_counter[source] += 1

        tags = row.get("tags") or []
        if isinstance(tags, list):
            for tag in tags:
                text = normalize_text(str(tag))
                if text:
                    tag_counter[text] += 1

    total_valid = len(rows) - len(invalid_rows)
    report = {
        "total_rows": len(rows),
        "valid_rows": total_valid,
        "invalid_rows": invalid_rows,
        "duplicate_user_assistant_rows": duplicate_user_assistant_rows,
        "duplicate_sample_rows": duplicate_sample_rows,
        "grounded_rows": grounded_rows,
        "refusal_rows": refusal_rows,
        "conflict_rows": conflict_rows,
        "grounded_ratio": round(grounded_rows / total_valid, 4) if total_valid else 0,
        "refusal_ratio": round(refusal_rows / total_valid, 4) if total_valid else 0,
        "conflict_ratio": round(conflict_rows / total_valid, 4) if total_valid else 0,
        "avg_system_chars": round(mean(system_lengths), 2) if system_lengths else 0,
        "avg_user_chars": round(mean(user_lengths), 2) if user_lengths else 0,
        "avg_assistant_chars": round(mean(assistant_lengths), 2) if assistant_lengths else 0,
        "top_scenarios": scenario_counter.most_common(15),
        "top_sources": source_counter.most_common(10),
        "top_tags": tag_counter.most_common(20),
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    """把关键摘要打印到终端。"""
    print("=== local llm dataset quality report ===")
    print(f"总行数: {report['total_rows']}")
    print(f"有效行数: {report['valid_rows']}")
    print(f"无效行数: {len(report['invalid_rows'])}")
    print(f"grounded 样本数: {report['grounded_rows']} ({report['grounded_ratio']})")
    print(f"refusal 样本数: {report['refusal_rows']} ({report['refusal_ratio']})")
    print(f"conflict 样本数: {report['conflict_rows']} ({report['conflict_ratio']})")
    print(f"平均 system 长度(字符): {report['avg_system_chars']}")
    print(f"平均 user 长度(字符): {report['avg_user_chars']}")
    print(f"平均 assistant 长度(字符): {report['avg_assistant_chars']}")
    print(f"user/assistant 重复行数: {len(report['duplicate_user_assistant_rows'])}")
    print(f"重复样本行数: {len(report['duplicate_sample_rows'])}")
    print(f"Top scenarios: {report['top_scenarios']}")
    print(f"Top tags: {report['top_tags'][:10]}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="检查本地生成模型 SFT 数据质量，帮助判断这批样本是否适合直接开训。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="待检查的本地生成模型训练集 JSONL 路径。",
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
