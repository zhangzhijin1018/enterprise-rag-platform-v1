"""导出 ONNX 模型。

这个脚本只负责把已经训练好的分类模型导出为 ONNX。
tokenizer 不在 ONNX 图内，所以仍然保留在 classifier 目录里。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _require_export_dependencies() -> dict[str, Any]:
    """延迟导入导出依赖。"""

    try:
        import torch
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("导出 ONNX 依赖缺失，请先安装 torch 和 transformers") from exc
    return {"torch": torch, "AutoModel": AutoModel}


def build_classifier_model_class(torch: Any, nn: Any, AutoModel: Any) -> Any:
    """构造和训练阶段同构的模型类。"""

    class BertWithDenseFeaturesClassifier(nn.Module):
        """训练与导出共用的模型定义。"""

        def __init__(self, encoder_name_or_path: str, dense_feature_dim: int, num_labels: int, dropout: float = 0.1) -> None:
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

        def forward(self, input_ids: Any, attention_mask: Any, dense_features: Any) -> Any:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = getattr(outputs, "pooler_output", None)
            if pooled_output is None:
                pooled_output = outputs.last_hidden_state[:, 0]
            dense_output = self.dense_encoder(dense_features)
            fused_output = self.dropout(nn.functional.normalize(torch.cat([pooled_output, dense_output], dim=1), dim=1))
            return self.classifier(fused_output)

    return BertWithDenseFeaturesClassifier


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="导出 ML 风控分类模型为 ONNX")
    parser.add_argument("--classifier-dir", type=str, required=True, help="分类模型目录")
    parser.add_argument("--onnx-output-path", type=str, required=True, help="ONNX 输出文件路径")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset 版本")
    return parser.parse_args()


def main() -> None:
    """导出入口。"""

    args = parse_args()
    deps = _require_export_dependencies()
    torch = deps["torch"]
    nn = torch.nn
    AutoModel = deps["AutoModel"]

    classifier_dir = Path(args.classifier_dir)
    manifest = json.loads((classifier_dir / "training_manifest.json").read_text(encoding="utf-8"))
    label_to_id = json.loads((classifier_dir / "label_to_id.json").read_text(encoding="utf-8"))

    BertWithDenseFeaturesClassifier = build_classifier_model_class(torch, nn, AutoModel)
    model = BertWithDenseFeaturesClassifier(
        encoder_name_or_path=manifest["encoder_path"],
        dense_feature_dim=int(manifest["dense_feature_dim"]),
        num_labels=len(label_to_id),
        dropout=float(manifest.get("dropout", 0.1)),
    )
    state_dict = torch.load(classifier_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    batch_size = 1
    seq_length = int(manifest["max_length"])
    dense_dim = int(manifest["dense_feature_dim"])
    dummy_input_ids = torch.ones((batch_size, seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    dummy_dense_features = torch.zeros((batch_size, dense_dim), dtype=torch.float32)

    output_path = Path(args.onnx_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_dense_features),
        str(output_path),
        input_names=["input_ids", "attention_mask", "dense_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "dense_features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=args.opset_version,
    )

    print(
        json.dumps(
            {
                "classifier_dir": str(classifier_dir),
                "onnx_output_path": str(output_path),
                "dense_feature_dim": dense_dim,
                "num_labels": len(label_to_id),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
