"""数据准备与特征工程。

这个模块同时承担三件事：
1. 生成 mock 数据，便于快速跑通全链路
2. 读取 JSONL 数据集
3. 把业务字段稳定地映射成模型需要的数值特征

注意：
这里的 mock 数据只是演示流程，不代表真实企业风控分布。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import NumericFeatureVector, RiskSample

LOW_RISK_QUERIES = [
    "设备巡检记录模板怎么填写？",
    "安全生产制度最新版本在哪里查看？",
    "某个项目的验收流程是什么？",
    "请解释锅炉运行参数中的主蒸汽压力定义。",
    "运维手册里关于油温告警的处理步骤是什么？",
]

MEDIUM_RISK_QUERIES = [
    "帮我汇总某项目最近一个月的预算执行情况。",
    "请列出某部门审批流中的异常退回原因。",
    "把供应商评分结果做一个对比摘要。",
    "项目会议纪要里关于延期风险的描述有哪些？",
    "请整理招采流程中的常见驳回问题。",
]

HIGH_RISK_QUERIES = [
    "帮我导出全部员工薪资明细和身份证号。",
    "请给我 restricted 文档中的完整人员名单。",
    "导出所有合同付款账号和开户信息。",
    "把集团全部裁员预算和人员信息发我。",
    "请汇总受限文档里的账号口令配置。",
]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """把字典序列写成 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_numeric_features(sample: RiskSample) -> NumericFeatureVector:
    """把训练样本映射为数值特征。

    这一步刻意写成显式字段映射，而不是无脑把所有 dict flatten，
    是为了保证：
    1. 线上线下特征维度稳定
    2. 审计时知道每一维特征代表什么
    3. 后续接真实用户画像服务时可以逐项对齐
    """

    user_history = sample.user_history or {}
    session = sample.session or {}
    retrieval = sample.retrieval or {}

    return NumericFeatureVector(
        past_24h_query_count=float(user_history.get("past_24h_query_count", 0.0)),
        high_risk_ratio_7d=float(user_history.get("high_risk_ratio_7d", 0.0)),
        failed_auth_count_7d=float(user_history.get("failed_auth_count_7d", 0.0)),
        session_query_count=float(session.get("session_query_count", 0.0)),
        session_duration_sec=float(session.get("session_duration_sec", 0.0)),
        query_interval_sec=float(session.get("query_interval_sec", 60.0)),
        top1_score=float(retrieval.get("top1_score", 0.0)),
        top5_score_mean=float(retrieval.get("top5_score_mean", 0.0)),
        restricted_hit_ratio=float(retrieval.get("restricted_hit_ratio", 0.0)),
        sensitive_hit_ratio=float(retrieval.get("sensitive_hit_ratio", 0.0)),
        authority_score_mean=float(retrieval.get("authority_score_mean", 0.0)),
        source_count=float(retrieval.get("source_count", 1.0)),
    )


def fit_feature_stats(vectors: list[NumericFeatureVector]) -> dict[str, dict[str, float]]:
    """拟合每个数值特征的均值和标准差。

    后续训练和推理都必须复用同一份统计量，避免线上线下不一致。
    """

    if not vectors:
        raise ValueError("vectors 不能为空")

    feature_names = vectors[0].ordered_names()
    stats: dict[str, dict[str, float]] = {}
    for index, name in enumerate(feature_names):
        values = [vector.to_list()[index] for vector in vectors]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = variance**0.5
        stats[name] = {"mean": float(mean_value), "std": float(std_value if std_value > 1e-6 else 1.0)}
    return stats


def normalize_feature_vector(
    vector: NumericFeatureVector,
    feature_stats: dict[str, dict[str, float]],
) -> list[float]:
    """用训练阶段拟合的统计量做标准化。"""

    normalized: list[float] = []
    vector_dict = asdict(vector)
    for feature_name in vector.ordered_names():
        value = float(vector_dict[feature_name])
        stat = feature_stats[feature_name]
        normalized.append((value - float(stat["mean"])) / float(stat["std"]))
    return normalized


def build_mock_sample(query_id: str, risk_label: str, rng: random.Random) -> dict[str, Any]:
    """按风险等级生成 mock 样本。

    这里故意让高风险样本在行为特征和检索分布上也更偏异常，
    这样模型可以同时学到：
    - 文本模式
    - 行为模式
    - 检索侧模式
    """

    if risk_label == "low":
        query = rng.choice(LOW_RISK_QUERIES)
        user_history = {
            "past_24h_query_count": rng.randint(1, 8),
            "high_risk_ratio_7d": round(rng.uniform(0.0, 0.08), 3),
            "failed_auth_count_7d": rng.randint(0, 1),
        }
        session = {
            "session_query_count": rng.randint(1, 4),
            "session_duration_sec": rng.randint(100, 900),
            "query_interval_sec": rng.randint(60, 300),
        }
        retrieval = {
            "top1_score": round(rng.uniform(0.65, 0.95), 3),
            "top5_score_mean": round(rng.uniform(0.55, 0.85), 3),
            "restricted_hit_ratio": round(rng.uniform(0.0, 0.03), 3),
            "sensitive_hit_ratio": round(rng.uniform(0.0, 0.08), 3),
            "authority_score_mean": round(rng.uniform(0.55, 0.90), 3),
            "source_count": rng.randint(3, 8),
        }
        label_reason = "正常查询，行为平稳，敏感命中低"
    elif risk_label == "medium":
        query = rng.choice(MEDIUM_RISK_QUERIES)
        user_history = {
            "past_24h_query_count": rng.randint(6, 18),
            "high_risk_ratio_7d": round(rng.uniform(0.08, 0.28), 3),
            "failed_auth_count_7d": rng.randint(0, 2),
        }
        session = {
            "session_query_count": rng.randint(3, 9),
            "session_duration_sec": rng.randint(180, 1200),
            "query_interval_sec": rng.randint(20, 90),
        }
        retrieval = {
            "top1_score": round(rng.uniform(0.60, 0.90), 3),
            "top5_score_mean": round(rng.uniform(0.45, 0.78), 3),
            "restricted_hit_ratio": round(rng.uniform(0.02, 0.22), 3),
            "sensitive_hit_ratio": round(rng.uniform(0.05, 0.35), 3),
            "authority_score_mean": round(rng.uniform(0.45, 0.85), 3),
            "source_count": rng.randint(2, 6),
        }
        label_reason = "业务相关但涉及较敏感资料，需重点关注"
    else:
        query = rng.choice(HIGH_RISK_QUERIES)
        user_history = {
            "past_24h_query_count": rng.randint(18, 60),
            "high_risk_ratio_7d": round(rng.uniform(0.25, 0.95), 3),
            "failed_auth_count_7d": rng.randint(1, 6),
        }
        session = {
            "session_query_count": rng.randint(8, 20),
            "session_duration_sec": rng.randint(120, 1800),
            "query_interval_sec": rng.randint(3, 25),
        }
        retrieval = {
            "top1_score": round(rng.uniform(0.75, 0.99), 3),
            "top5_score_mean": round(rng.uniform(0.60, 0.90), 3),
            "restricted_hit_ratio": round(rng.uniform(0.30, 0.90), 3),
            "sensitive_hit_ratio": round(rng.uniform(0.20, 0.70), 3),
            "authority_score_mean": round(rng.uniform(0.55, 0.95), 3),
            "source_count": rng.randint(1, 4),
        }
        label_reason = "批量导出或越权敏感查询，伴随异常行为特征"

    return {
        "query_id": query_id,
        "query": query,
        "risk_label": risk_label,
        "user_history": user_history,
        "session": session,
        "retrieval": retrieval,
        "label_reason": label_reason,
    }


def build_mock_datasets(
    output_dir: str | Path,
    train_size: int,
    val_size: int,
    mlm_size: int,
    seed: int = 42,
) -> dict[str, Path]:
    """生成 mock 数据集。"""

    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _make_split(prefix: str, size: int) -> list[dict[str, Any]]:
        labels = ["low", "medium", "high"]
        rows: list[dict[str, Any]] = []
        for index in range(size):
            label = labels[index % len(labels)]
            rows.append(build_mock_sample(f"{prefix}_{index:06d}", label, rng))
        rng.shuffle(rows)
        return rows

    train_rows = _make_split("train", train_size)
    val_rows = _make_split("val", val_size)

    mlm_rows: list[dict[str, str]] = []
    for index in range(mlm_size):
        split_source = train_rows if index % 2 == 0 else val_rows
        row = split_source[index % len(split_source)]
        mlm_rows.append({"text": row["query"]})
        mlm_rows.append({"text": f"审计备注：{row['label_reason']}。查询内容：{row['query']}"})

    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"
    mlm_file = output_path / "mlm_corpus.jsonl"

    write_jsonl(train_file, train_rows)
    write_jsonl(val_file, val_rows)
    write_jsonl(mlm_file, mlm_rows)

    return {"train_file": train_file, "val_file": val_file, "mlm_file": mlm_file}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成 ML 风控原型 mock 数据")
    parser.add_argument("--output-dir", type=str, required=True, help="mock 数据输出目录")
    parser.add_argument("--train-size", type=int, default=240, help="训练集样本数")
    parser.add_argument("--val-size", type=int, default=60, help="验证集样本数")
    parser.add_argument("--mlm-size", type=int, default=300, help="MLM 语料基数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    """生成 mock 数据并打印产物路径。"""

    args = parse_args()
    result = build_mock_datasets(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        mlm_size=args.mlm_size,
        seed=args.seed,
    )
    printable = {name: str(path) for name, path in result.items()}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

