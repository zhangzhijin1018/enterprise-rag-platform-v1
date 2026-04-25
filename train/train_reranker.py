"""`bge-reranker-large` 微调脚本。

这份脚本专门服务于当前企业 RAG 项目的“二阶段精排”问题，目标不是让模型更会聊天，
而是让它更擅长判断：

1. 同一个 query 下，哪个 chunk 更像真正证据；
2. 哪些 chunk 虽然词面相近，但不应该排在前面；
3. 当召回链路已经把候选找出来后，如何把正确证据稳定推到 top-3 / top-5。

为什么这一步通常比先调生成模型更值：

- 当前项目已经有 native hybrid search、metadata_filter、metadata_boost 和 governance；
- 很多 badcase 的根因不是“没召回”，而是“召回到了但顺序不准”；
- reranker 微调的收益最容易通过 `/eval`、badcase 回放和引用正确率看到。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.utils.data import DataLoader


logger = logging.getLogger("train_reranker")


@dataclass
class TrainConfig:
    """训练参数配置。

    字段说明：
    - model_name:
      基础 reranker 模型路径。默认指向本地 `./modes/bge-reranker-large`，
      这样训练时不依赖网络下载，也更符合当前仓库本地模型优先的使用方式。
    - train_path:
      训练集 JSONL 路径。每行至少要有 `query`、`positive` 和 `negative/negatives`。
    - output_dir:
      微调后模型输出目录。训练结束后会把模型权重和 `run_manifest.json` 一起写到这里。
    - dev_path:
      可选验证集路径。提供后会在训练过程中定期评估，并保存验证集最优模型。
    - batch_size:
      单卡 batch size。默认 `8` 是在“文本对长度可达 512、又希望普通单卡能跑”的
      情况下相对稳妥的起点；如果显存较小可降到 `4/2`，显存较大可以继续加。
    - epochs:
      训练轮数。默认 `2` 是因为 reranker 在企业场景第一轮微调时更怕过拟合而不是不收敛，
      所以先用小 epoch 观察 `/eval` 和 badcase 变化更稳。
    - learning_rate:
      学习率。默认 `2e-5` 是 CrossEncoder 微调里比较稳的常用值；
      太大容易把原始通用排序能力破坏掉，太小则第一轮看不到明显收益。
    - warmup_ratio:
      预热比例。默认 `0.1` 是 Transformer 微调的经典保守值，
      可以减轻训练前期梯度波动过大的问题。
    - max_length:
      query + chunk 的最大 token 长度。默认 `512` 是一个平衡点：
      足够覆盖大多数制度段落和 SOP 片段，又不会让显存成本过高。
    - seed:
      随机种子。用于让抽样和训练更可复现，便于你做 badcase 对比。
    - eval_steps:
      多久在 dev 集上评估一次。默认 `200` 更适合几千到几万样本规模的第一轮实验。
    - save_best_model:
      只有在提供 dev 集时才真正有意义。默认保留最优模型，避免只拿最后一个 checkpoint。
    - train_max_rows:
      只在试跑或 smoke test 时使用，方便先用前几百条样本验证训练链路是否打通。
    """

    model_name: str
    train_path: str
    output_dir: str
    dev_path: str | None = None
    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_length: int = 512
    seed: int = 42
    eval_steps: int = 200
    save_best_model: bool = True
    train_max_rows: int | None = None


def configure_logging() -> None:
    """配置训练日志。

    这里故意保持日志格式简单：
    - 时间
    - 级别
    - logger 名
    - 消息

    原因是训练日志最常用于：
    1. 看当前跑到哪一步；
    2. 回头核对训练超参数和数据规模；
    3. 排查 OOM、路径错误、样本为空等最常见问题。
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def set_seed(seed: int) -> None:
    """设置随机种子。

    这里先做最关键的一层 `random.seed`，目的是：
    - 保证样本顺序扰动尽量可复现；
    - 方便你多轮训练时对比“同一批数据、不同参数”。

    之所以没有在这里额外硬塞 `numpy` 或 `torch` 的全量随机种子，是因为当前脚本
    优先追求最小依赖和最小改动；如果后面你做严格实验复现，再补全 `torch.manual_seed`
    也很自然。
    """

    random.seed(seed)


def load_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """读取 JSONL 文件。

    这里默认把一行视为一个“训练样本对象”，而不是预先假定固定 schema。
    这样做的好处是：
    - 前期你可以不断扩充字段，比如 `scenario`、`tags`、`source`；
    - 主训练逻辑只关心真正必要的 `query/positive/negative(s)`；
    - 其余字段可以保留给后续分析、抽样和训练后溯源。
    """

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no training rows found in {path}")
    return rows


def normalize_triplet_row(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    """把不同写法统一成 `query / positive / negatives`。

    当前训练数据允许两种负样本表达：
    - `negative`: 单个主负例
    - `negatives`: 额外负例列表

    这么设计是为了兼容两种常见数据来源：
    1. 人工标注 triplet：通常只有一个最主要的负例；
    2. 自动抽样数据：通常会带 1 个主负例 + 若干补充负例。

    返回值说明：
    - query:
      用户问题或检索查询。
    - positive:
      该 query 对应的正确 chunk / 正确证据文本。
    - negatives:
      一个负例列表。列表越丰富，pairwise 训练时能看到的“错候选”模式越多。
    """

    query = str(row.get("query", "")).strip()
    positive = str(row.get("positive", "")).strip()
    negative = row.get("negative")
    negatives = row.get("negatives")

    if not query or not positive:
        raise ValueError(f"invalid row, missing query/positive: {row}")

    all_negatives: list[str] = []
    if isinstance(negative, str) and negative.strip():
        all_negatives.append(negative.strip())
    if isinstance(negatives, list):
        for item in negatives:
            text = str(item).strip()
            if text:
                all_negatives.append(text)

    if not all_negatives:
        raise ValueError(f"invalid row, missing negative/negatives: {row}")

    return query, positive, all_negatives


def build_pairwise_examples(rows: list[dict[str, Any]]) -> list[InputExample]:
    """把 triplet 数据转换成 CrossEncoder 的 pairwise 样本。

    核心原理：
    - 当前项目的 reranker 是 `CrossEncoder`；
    - `CrossEncoder` 的输入不是单独的 query 或 chunk embedding，
      而是“query + chunk”成对喂给模型；
    - 模型会直接输出这一对文本的相关性得分。

    因此训练时最直接、最稳的做法就是：
    - 正例对 `(query, positive)` 标成 `1.0`
    - 负例对 `(query, negative)` 标成 `0.0`

    为什么不在这里直接做 listwise 排序训练：
    - 第一轮落地要优先简单、稳定、好复现；
    - 当前仓库本来就基于 `CrossEncoder` 推理，pairwise 训练和线上形式最一致；
    - 当你先把第一轮精排收益跑出来后，再考虑更复杂的 ranking loss 才更划算。
    """

    examples: list[InputExample] = []
    for row in rows:
        query, positive, negatives = normalize_triplet_row(row)
        examples.append(InputExample(texts=[query, positive], label=1.0))
        for neg in negatives:
            examples.append(InputExample(texts=[query, neg], label=0.0))
    return examples


def build_binary_evaluator(rows: list[dict[str, Any]], *, name: str) -> CEBinaryClassificationEvaluator:
    """构建验证集 evaluator。

    这里用的是 `CEBinaryClassificationEvaluator`，原因是当前训练样本本质上就是：
    - 正例对：1
    - 负例对：0

    它不是最终线上排序指标，但很适合作为第一轮训练时的稳定校验信号。
    更高层的真实收益，后面仍然要通过：
    - `/eval`
    - badcase 回放
    - top-k 命中率
    - 引用正确率
    来判断。
    """

    sentence_pairs: list[list[str]] = []
    labels: list[int] = []
    for row in rows:
        query, positive, negatives = normalize_triplet_row(row)
        sentence_pairs.append([query, positive])
        labels.append(1)
        for neg in negatives:
            sentence_pairs.append([query, neg])
            labels.append(0)
    return CEBinaryClassificationEvaluator(
        sentence_pairs=sentence_pairs,
        labels=labels,
        name=name,
    )


def save_run_manifest(cfg: TrainConfig, train_rows: int, train_pairs: int, dev_rows: int | None) -> None:
    """把关键训练参数落盘。

    这一步看起来像小事，但在真实训练里很重要。

    `run_manifest.json` 解决的是两个常见问题：
    1. 过一周后你已经忘了当时到底用了哪批数据、多少 epoch、什么学习率；
    2. 你手上有多个微调版本，但很难对应回训练配置。

    当前落盘字段说明：
    - created_at: 训练启动时间
    - config: 本次训练完整参数快照
    - train_rows: 原始 triplet 行数
    - train_pairs: 真正送进 CrossEncoder 的 pairwise 样本数
    - dev_rows: 验证集 triplet 行数
    """

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "train_rows": train_rows,
        "train_pairs": train_pairs,
        "dev_rows": dev_rows,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> TrainConfig:
    """解析命令行参数。

    这里尽量保持“完整但不花哨”的 CLI 设计，目的是：
    - 你可以直接复制命令跑；
    - 参数含义和 `TrainConfig` 一一对应；
    - 新人接手时不用再猜某个值会影响哪一步。
    """

    parser = argparse.ArgumentParser(
        description="对企业 RAG 场景的 `bge-reranker-large` 做第一轮精排微调。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="./modes/bge-reranker-large",
        help="基础 reranker 模型路径。通常指向本地原始模型目录。",
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="训练集 JSONL 路径。每行至少需要 `query`、`positive` 和 `negative/negatives`。",
    )
    parser.add_argument(
        "--dev-path",
        default=None,
        help="可选验证集 JSONL 路径。提供后会周期性评估并保存验证集最优模型。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="训练输出目录，训练好的模型和 `run_manifest.json` 会写到这里。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="单卡 batch size。显存不够时先降到 4 或 2，再考虑调其他参数。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="训练轮数。企业 reranker 第一轮建议先小 epoch 试跑，避免过拟合。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="优化器学习率。`2e-5` 是 CrossEncoder 微调中较稳的常用起点。",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="学习率预热比例。默认 10%%，用于降低训练初期梯度波动。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="query 和 chunk 拼接后的最大 token 长度。更大通常更吃显存。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于让训练顺序和实验结果尽量可复现。",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="当提供 dev 集时，每隔多少个训练 step 做一次验证。",
    )
    parser.add_argument(
        "--train-max-rows",
        type=int,
        default=None,
        help="可选，只取前 N 行训练数据，常用于 smoke test 或小样本试跑。",
    )
    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        train_path=args.train_path,
        dev_path=args.dev_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        seed=args.seed,
        eval_steps=args.eval_steps,
        train_max_rows=args.train_max_rows,
    )


def main() -> None:
    """训练主入口。

    主流程可以概括成 6 步：
    1. 读配置和训练数据；
    2. 把 triplet 转成 pairwise 样本；
    3. 可选加载 dev 集 evaluator；
    4. 落盘 `run_manifest.json`；
    5. 加载基础 reranker 模型；
    6. 调 `model.fit(...)` 完成微调。

    这里故意把“数据准备”和“模型训练”分成非常清楚的两段，是为了后续方便你：
    - 单独检查样本质量；
    - 单独扩展自动抽样脚本；
    - 单独替换基础模型，而不打乱训练入口。
    """

    configure_logging()
    cfg = parse_args()
    set_seed(cfg.seed)

    # 先读 triplet，再转换成 pairwise，是因为 triplet 更接近标注和 badcase 数据的自然形态；
    # 真正喂给 CrossEncoder 时再展开成 pairwise，可以兼顾“数据好管理”和“训练好兼容”。
    logger.info("loading training rows from %s", cfg.train_path)
    train_rows = load_jsonl(cfg.train_path, limit=cfg.train_max_rows)
    train_examples = build_pairwise_examples(train_rows)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size)
    logger.info("train rows=%d, train pairwise examples=%d", len(train_rows), len(train_examples))

    dev_rows: list[dict[str, Any]] | None = None
    evaluator = None
    if cfg.dev_path:
        # 只有显式提供 dev 集时才启用 evaluator。
        # 这是因为很多第一轮实验先只关注“链路能不能跑通”，未必一开始就有干净 dev 集。
        logger.info("loading dev rows from %s", cfg.dev_path)
        dev_rows = load_jsonl(cfg.dev_path)
        evaluator = build_binary_evaluator(dev_rows, name="enterprise-rag-reranker-dev")
        logger.info("dev rows=%d", len(dev_rows))

    save_run_manifest(
        cfg,
        train_rows=len(train_rows),
        train_pairs=len(train_examples),
        dev_rows=len(dev_rows) if dev_rows else None,
    )

    logger.info("loading base model from %s", cfg.model_name)
    model = CrossEncoder(
        cfg.model_name,
        num_labels=1,
        max_length=cfg.max_length,
    )

    # warmup_steps 的目的，是在训练初期让学习率从较小值平滑升起，减小梯度震荡。
    # 对于基于预训练模型做微调的 reranker，这一步通常能让训练更稳，尤其是数据量不大时。
    warmup_steps = int(len(train_loader) * cfg.epochs * cfg.warmup_ratio)
    logger.info(
        "start training: epochs=%d batch_size=%d lr=%s warmup_steps=%d",
        cfg.epochs,
        cfg.batch_size,
        cfg.learning_rate,
        warmup_steps,
    )

    # `CrossEncoder.fit` 会直接处理：
    # - 前向
    # - loss 计算
    # - 反向传播
    # - evaluator 触发
    # - best model 保存
    #
    # 这里我们尽量只传最关键、最容易解释清楚的参数，避免把脚本写成“大量开关但没人敢改”的状态。
    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        epochs=cfg.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": cfg.learning_rate},
        output_path=cfg.output_dir,
        evaluation_steps=cfg.eval_steps if evaluator is not None else 0,
        save_best_model=cfg.save_best_model if evaluator is not None else True,
        show_progress_bar=True,
    )

    logger.info("training finished, saved to %s", cfg.output_dir)
    logger.info(
        "next step: set RERANKER_MODEL_NAME=%s and run /eval + badcase replay",
        cfg.output_dir,
    )


if __name__ == "__main__":
    main()
