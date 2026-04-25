"""BERT 风控模型训练脚本。

这个脚本实现两个阶段：
1. 继续预训练（MLM）
2. 风控分类微调（文本 + 数值特征融合）

为什么拆成两段：
- MLM 负责让模型先适应企业内部查询语料分布
- 分类微调负责把“风险等级”这个监督信号真正学进去
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .data_pipeline import build_numeric_features, fit_feature_stats, load_jsonl, normalize_feature_vector
from .schemas import NumericFeatureVector, RiskSample


def _require_training_dependencies() -> dict[str, Any]:
    """延迟导入训练依赖。

    这样即使当前环境还没安装 transformers / datasets，也不会影响查看代码和语法检查。
    真正运行训练时，再给出明确错误信息。
    """

    try:
        import torch
        import torch.nn as nn
        from datasets import Dataset
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModel,
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "训练依赖缺失，请先安装 transformers / datasets / torch / accelerate / evaluate"
        ) from exc

    return {
        "torch": torch,
        "nn": nn,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "AdamW": AdamW,
        "AutoModel": AutoModel,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


class RiskClassificationDataset:
    """分类训练数据集。

    每个样本包含三类输入：
    - `input_ids`
    - `attention_mask`
    - `dense_features`

    其中 dense_features 就是行为特征 + session 特征 + 检索分布特征。
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        feature_stats: dict[str, dict[str, float]],
        label_to_id: dict[str, int],
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.feature_stats = feature_stats
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample = RiskSample(**row)
        tokenized = self.tokenizer(
            sample.query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        vector = build_numeric_features(sample)
        normalized_features = normalize_feature_vector(vector, self.feature_stats)
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "dense_features": normalized_features,
            "labels": self.label_to_id[sample.risk_label],
        }


def collate_classification_batch(batch: list[dict[str, Any]], torch: Any) -> dict[str, Any]:
    """分类 batch 拼装函数。"""

    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "dense_features": torch.tensor([item["dense_features"] for item in batch], dtype=torch.float32),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
    }


def build_classifier_model_class(torch: Any, nn: Any, AutoModel: Any) -> Any:
    """动态构造分类模型类。

    这里用工厂函数，是为了减少模块级强依赖，避免未安装 torch/transformers 时导入失败。
    """

    class BertWithDenseFeaturesClassifier(nn.Module):
        """文本编码 + 数值特征融合分类模型。

        前向流程：
        1. 文本走 BERT，取 `[CLS]` 或 pooler 输出
        2. 数值特征走两层 MLP
        3. 文本向量和数值向量拼接
        4. 输出 low / medium / high 三分类 logits
        """

        def __init__(
            self,
            encoder_name_or_path: str,
            dense_feature_dim: int,
            num_labels: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
            hidden_size = int(self.encoder.config.hidden_size)
            self.dense_encoder = nn.Sequential(
                nn.Linear(dense_feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size + 32, num_labels)
            self.loss_fn = nn.CrossEntropyLoss()

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            dense_features: Any,
            labels: Any | None = None,
        ) -> dict[str, Any]:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = getattr(encoder_outputs, "pooler_output", None)
            if pooled_output is None:
                pooled_output = encoder_outputs.last_hidden_state[:, 0]
            dense_output = self.dense_encoder(dense_features)
            fused_output = self.dropout(nn.functional.normalize(torch.cat([pooled_output, dense_output], dim=1), dim=1))
            logits = self.classifier(fused_output)
            loss = self.loss_fn(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits}

    return BertWithDenseFeaturesClassifier


def run_mlm_training(args: argparse.Namespace, deps: dict[str, Any]) -> None:
    """执行继续预训练（MLM）。"""

    Dataset = deps["Dataset"]
    AutoModelForMaskedLM = deps["AutoModelForMaskedLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    DataCollatorForLanguageModeling = deps["DataCollatorForLanguageModeling"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    mlm_rows = load_jsonl(args.mlm_train_file)
    if not mlm_rows:
        raise ValueError("MLM 训练集为空")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    texts = [str(row["text"]) for row in mlm_rows if row.get("text")]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    model = AutoModelForMaskedLM.from_pretrained(args.base_model_name)
    output_dir = Path(args.mlm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.mlm_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    manifest = {
        "stage": "mlm",
        "base_model_name": args.base_model_name,
        "max_length": args.max_length,
        "num_rows": len(texts),
        "output_dir": str(output_dir),
    }
    (output_dir / "mlm_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_predictions(predictions: list[int], labels: list[int]) -> dict[str, float]:
    """计算简单分类指标。

    这里只实现最核心的 accuracy，避免为了实验原型引入额外复杂度。
    如果后续要接入正式评测，建议扩展 high risk precision / recall / F1。
    """

    if not labels:
        return {"accuracy": 0.0}
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return {"accuracy": round(correct / len(labels), 6)}


def save_classifier_artifacts(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    feature_stats: dict[str, dict[str, float]],
    label_to_id: dict[str, int],
    training_manifest: dict[str, Any],
    torch: Any,
) -> None:
    """保存分类模型产物。

    这里刻意把推理需要的元信息都保存下来，避免 ONNX 或线上服务二次猜测。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    (output_dir / "feature_stats.json").write_text(
        json.dumps(feature_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_to_id.json").write_text(
        json.dumps(label_to_id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_classification_training(args: argparse.Namespace, deps: dict[str, Any]) -> None:
    """执行风控分类微调。"""

    torch = deps["torch"]
    nn = deps["nn"]
    DataLoader = deps["DataLoader"]
    AdamW = deps["AdamW"]
    AutoTokenizer = deps["AutoTokenizer"]
    AutoModel = deps["AutoModel"]

    train_rows = load_jsonl(args.train_file)
    val_rows = load_jsonl(args.val_file)
    if not train_rows:
        raise ValueError("train_file 为空")
    if not val_rows:
        raise ValueError("val_file 为空")

    label_to_id = {"low": 0, "medium": 1, "high": 2}
    id_to_label = {value: key for key, value in label_to_id.items()}

    train_vectors = [build_numeric_features(RiskSample(**row)) for row in train_rows]
    feature_stats = fit_feature_stats(train_vectors)
    dense_feature_dim = len(train_vectors[0].to_list())

    encoder_path = args.domain_model_dir or args.base_model_name
    tokenizer_source = Path(encoder_path) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source) if tokenizer_source.exists() else encoder_path)

    dataset_cls = RiskClassificationDataset
    train_dataset = dataset_cls(train_rows, tokenizer, args.max_length, feature_stats, label_to_id)
    val_dataset = dataset_cls(val_rows, tokenizer, args.max_length, feature_stats, label_to_id)

    BertWithDenseFeaturesClassifier = build_classifier_model_class(torch, nn, AutoModel)
    model = BertWithDenseFeaturesClassifier(
        encoder_name_or_path=encoder_path,
        dense_feature_dim=dense_feature_dim,
        num_labels=len(label_to_id),
        dropout=args.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_classification_batch(batch, torch),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_classification_batch(batch, torch),
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.cls_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_accuracy = -1.0
    best_state_dict: dict[str, Any] | None = None
    global_step = 0

    for epoch in range(args.cls_epochs):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs["loss"]
            if loss is None:
                raise RuntimeError("分类训练阶段 loss 不应为空")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            train_loss_sum += float(loss.detach().cpu().item())
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(
                    json.dumps(
                        {
                            "stage": "cls_train",
                            "epoch": epoch + 1,
                            "step": global_step,
                            "loss": round(float(loss.detach().cpu().item()), 6),
                        },
                        ensure_ascii=False,
                    )
                )

        model.eval()
        predictions: list[int] = []
        labels: list[int] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                outputs = model(**batch)
                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.detach().cpu().tolist())
                labels.extend(batch["labels"].detach().cpu().tolist())

        metrics = evaluate_predictions(predictions, labels)
        epoch_loss = train_loss_sum / max(1, len(train_loader))
        print(
            json.dumps(
                {
                    "stage": "cls_eval",
                    "epoch": epoch + 1,
                    "train_loss": round(epoch_loss, 6),
                    **metrics,
                },
                ensure_ascii=False,
            )
        )
        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is None:
        raise RuntimeError("训练未产生可保存的最佳模型")

    model.load_state_dict(best_state_dict)
    output_dir = Path(args.classifier_output_dir)
    training_manifest = {
        "stage": "classification",
        "base_model_name": args.base_model_name,
        "encoder_path": encoder_path,
        "max_length": args.max_length,
        "dense_feature_dim": dense_feature_dim,
        "labels": id_to_label,
        "best_accuracy": best_accuracy,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "feature_names": NumericFeatureVector().ordered_names(),
        "dropout": args.dropout,
    }
    save_classifier_artifacts(output_dir, model, tokenizer, feature_stats, label_to_id, training_manifest, torch)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 ML 风控模型")
    parser.add_argument("--stage", choices=["mlm", "cls", "all"], required=True, help="训练阶段")
    parser.add_argument("--base-model-name", type=str, required=True, help="基础中文 BERT 模型名或本地路径")
    parser.add_argument("--domain-model-dir", type=str, default="", help="继续预训练后的模型目录；为空时直接使用 base model")
    parser.add_argument("--mlm-train-file", type=str, default="", help="MLM 训练集 JSONL")
    parser.add_argument("--train-file", type=str, default="", help="分类训练集 JSONL")
    parser.add_argument("--val-file", type=str, default="", help="分类验证集 JSONL")
    parser.add_argument("--mlm-output-dir", type=str, default="", help="MLM 模型输出目录")
    parser.add_argument("--classifier-output-dir", type=str, default="", help="分类模型输出目录")
    parser.add_argument("--max-length", type=int, default=128, help="token 最大长度")
    parser.add_argument("--mlm-epochs", type=int, default=1, help="继续预训练 epoch 数")
    parser.add_argument("--cls-epochs", type=int, default=3, help="分类训练 epoch 数")
    parser.add_argument("--train-batch-size", type=int, default=8, help="训练 batch size")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="验证 batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="warmup 比例")
    parser.add_argument("--logging-steps", type=int, default=10, help="日志步长")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用 CPU 训练")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """校验训练参数。"""

    if args.stage in {"mlm", "all"} and not args.mlm_train_file:
        raise ValueError("执行 mlm/all 时必须提供 --mlm-train-file")
    if args.stage in {"mlm", "all"} and not args.mlm_output_dir:
        raise ValueError("执行 mlm/all 时必须提供 --mlm-output-dir")
    if args.stage in {"cls", "all"} and not args.train_file:
        raise ValueError("执行 cls/all 时必须提供 --train-file")
    if args.stage in {"cls", "all"} and not args.val_file:
        raise ValueError("执行 cls/all 时必须提供 --val-file")
    if args.stage in {"cls", "all"} and not args.classifier_output_dir:
        raise ValueError("执行 cls/all 时必须提供 --classifier-output-dir")


def main() -> None:
    """训练入口。"""

    args = parse_args()
    validate_args(args)
    deps = _require_training_dependencies()

    if args.stage in {"mlm", "all"}:
        run_mlm_training(args, deps)

    if args.stage in {"cls", "all"}:
        # 如果执行 all，并且未显式传入 domain_model_dir，则自动使用 mlm 输出目录继续做分类训练。
        if args.stage == "all" and not args.domain_model_dir:
            args.domain_model_dir = args.mlm_output_dir
        run_classification_training(args, deps)


if __name__ == "__main__":
    main()
