"""BGEM3 检索编码后端。

这个模块把 `BGEM3EmbeddingFunction` 的加载和推理细节单独收口，
让上层 dense / sparse 检索器只关心：

- 能不能拿到 dense 向量
- 能不能拿到 sparse 向量

而不需要在多个地方重复处理：
- 依赖缺失
- 设备选择
- 模型懒加载
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from core.config.settings import Settings, get_settings
from core.observability import get_logger

logger = get_logger(__name__)


def sparse_row_to_milvus_dict(row: Any) -> dict[int, float]:
    """把单条 BGEM3 sparse 向量转成 Milvus `SPARSE_FLOAT_VECTOR` 可接受的格式。

    当前 BGEM3 返回的 sparse 结果通常是 scipy sparse row，例如：
    - `csr_array`
    - `coo_array`

    Milvus 写入与查询时更稳妥的输入是：
    - `{index: value}` 这样的稀疏字典

    因此这里统一做一次格式收敛，避免这部分逻辑散落在：
    - 入库落 Milvus
    - sparse 查询
    两个地方重复实现。
    """

    if row is None:
        return {}
    if isinstance(row, dict):
        if "indices" in row and "values" in row:
            return {
                int(index): float(value)
                for index, value in zip(row["indices"], row["values"], strict=True)
            }
        return {int(index): float(value) for index, value in row.items()}

    to_coo = getattr(row, "tocoo", None)
    if callable(to_coo):
        coo = to_coo()
        return {
            int(index): float(value)
            for index, value in zip(coo.col.tolist(), coo.data.tolist(), strict=True)
        }

    if isinstance(row, np.ndarray):
        flat = row.reshape(-1)
        return {
            int(index): float(value)
            for index, value in enumerate(flat.tolist())
            if float(value) != 0.0
        }

    raise TypeError(f"unsupported BGEM3 sparse row type: {type(row)!r}")


def _resolve_device(device: str) -> str:
    """把 `auto` 设备解析成可实际使用的设备名。"""

    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - 这里只做尽力判断
        pass
    return "cpu"


def _has_safetensors_weights(model_name: str) -> bool:
    """判断本地模型目录中是否存在 safetensors 权重文件。"""

    model_path = Path(model_name)
    if not model_path.exists() or not model_path.is_dir():
        return False
    return any(model_path.glob("*.safetensors"))


@lru_cache(maxsize=4)
def _load_bgem3_function(
    model_name: str,
    device: str,
    use_fp16: bool,
) -> Any | None:
    """懒加载 BGEM3EmbeddingFunction。

    返回 `None` 表示当前环境不具备 BGEM3 运行条件，此时上层检索器会显式降级。
    """

    try:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
    except ModuleNotFoundError:
        logger.warning(
            "BGEM3 backend unavailable because pymilvus.model is missing; fallback to classic retrievers",
            extra={"event": "bgem3_unavailable", "model_name": model_name},
        )
        return None
    try:
        return BGEM3EmbeddingFunction(
            model_name=model_name,
            use_fp16=use_fp16,
            device=device,
        )
    except Exception as exc:  # pragma: no cover - 依赖真实模型环境
        message = str(exc)
        extra = {
            "event": "bgem3_init_failed",
            "model_name": model_name,
            "device": device,
            "error": message,
        }
        if "upgrade torch to at least v2.6" in message and not _has_safetensors_weights(model_name):
            logger.warning(
                "BGEM3 backend blocked by local environment: torch<2.6 cannot load pytorch_model.bin; "
                "upgrade torch or replace the model with a safetensors variant, fallback to classic retrievers",
                extra=extra,
            )
            return None
        logger.warning(
            "BGEM3 backend initialization failed; fallback to classic retrievers",
            extra=extra,
        )
        return None


class BGEM3Backend:
    """BGEM3 编码器包装层。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._device = _resolve_device(self._settings.bgem3_device)

    @property
    def enabled(self) -> bool:
        """当前配置是否要求优先使用 BGEM3。"""

        return self._settings.retrieval_embedding_backend == "bgem3"

    def get_function(self) -> Any | None:
        """返回 BGEM3EmbeddingFunction 实例；不可用时返回 None。"""

        if not self.enabled:
            return None
        return _load_bgem3_function(
            self._settings.embedding_model_name,
            self._device,
            self._settings.bgem3_use_fp16,
        )

    def encode_documents(self, texts: list[str]) -> dict[str, Any]:
        """编码文档文本，返回 BGEM3 原生输出结构。"""

        fn = self.get_function()
        if fn is None:
            raise RuntimeError("BGEM3 backend is not available in the current environment")
        return fn.encode_documents(texts)

    def encode_queries(self, texts: list[str]) -> dict[str, Any]:
        """编码查询文本，返回 BGEM3 原生输出结构。"""

        fn = self.get_function()
        if fn is None:
            raise RuntimeError("BGEM3 backend is not available in the current environment")
        return fn.encode_queries(texts)
