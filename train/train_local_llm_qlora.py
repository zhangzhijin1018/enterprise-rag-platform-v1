"""本地生成式模型 QLoRA/SFT 微调脚本。

这份脚本面向当前企业 RAG 项目的“本地生成模型对齐”场景，核心目标不是训练通用聊天模型，
而是让本地模型在以下任务上更稳：

1. 基于证据回答（grounded answer）；
2. 证据不足时明确拒答；
3. 多文档冲突时输出保守、可解释的说明；
4. 输出更符合企业风格、带引用占位或结构化段落。

适用边界：
- 当前更适合在“检索链路已经相对稳定”之后再做；
- 推荐作为高敏场景 `local_only / restricted` 的第二阶段能力；
- 默认假设基础模型为 `Qwen/Qwen3-14B-Instruct` 或同量级 instruct 模型。

注意：
- 这份脚本设计目标是“可迁移到 Linux + CUDA 训练机直接使用”；
- 当前 macOS 开发机适合做脚本整理、样例数据准备和命令校验，
  不建议直接在本机执行真实 QLoRA 训练。
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


logger = logging.getLogger("train_local_llm_qlora")


@dataclass
class TrainConfig:
    """本地生成模型微调配置。

    字段说明：
    - model_name:
      基础生成模型目录或 Hugging Face 模型名。当前建议指向本地 `Qwen3-14B-Instruct`。
    - train_path:
      训练集 JSONL 路径。推荐每行使用 `messages` 结构。
    - output_dir:
      LoRA/QLoRA 适配器输出目录。
    - dev_path:
      可选验证集。当前脚本先保留参数，默认第一轮不强依赖自动 evaluator。
    - max_seq_length:
      单条样本最大序列长度。默认 `4096` 更适合 grounded answer、拒答和冲突解释场景。
    - per_device_train_batch_size:
      单卡 batch size。QLoRA 下生成模型显存压力大，通常从 `1` 起步最稳。
    - gradient_accumulation_steps:
      梯度累计步数。通过“小 batch + 累计”获得更可控的有效 batch。
    - learning_rate:
      LoRA 微调学习率。默认 `2e-4` 是 QLoRA/SFT 比较常见的稳妥起点。
    - num_train_epochs:
      训练轮数。默认 `2` 先观察质量和过拟合迹象。
    - logging_steps:
      日志输出间隔。
    - save_steps:
      checkpoint 保存间隔。
    - lora_r / lora_alpha / lora_dropout:
      LoRA 核心超参数。默认是一组偏稳、社区常用的起点。
    - load_in_4bit:
      是否启用 4bit 量化加载。QLoRA 的核心就是在低显存前提下做可训练微调。
    - seed:
      随机种子。
    - train_max_rows:
      只在 smoke test 或样例调试时使用。
    """

    model_name: str
    train_path: str
    output_dir: str
    dev_path: str | None = None
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    logging_steps: int = 10
    save_steps: int = 200
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    seed: int = 42
    train_max_rows: int | None = None


def configure_logging() -> None:
    """配置训练日志。

    这里刻意保持最基础的日志格式，不引入额外日志框架，原因有两个：

    1. 这类训练脚本最常见的问题不是复杂业务错误，而是：
       - 模型路径不对
       - 训练集为空
       - 量化依赖没装
       - 显存不足
    2. 在真实训练机上，训练日志通常会被：
       - `tee`
       - 任务调度平台
       - notebook / tmux
       捕获，所以简单统一格式最利于排查。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def set_seed(seed: int) -> None:
    """设置最基本的随机种子。

    当前先只设置 `random.seed`，目的是让：
    - 数据顺序扰动
    - 某些 Python 侧随机行为
    尽量可复现。

    这里没有直接把 `torch.cuda.manual_seed_all(...)`、`numpy.random.seed(...)`
    一股脑全塞进来，是因为这份脚本当前优先追求：
    - 依赖尽量少
    - 先把训练链跑通
    - 代码在没有训练依赖的开发机上也能先过语法和 CLI

    如果后面进入严格实验复现阶段，再补全深一点的随机性控制会更合适。
    """
    random.seed(seed)


def load_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """读取 JSONL 数据。

    为什么训练集采用 JSONL：
    - 逐行追加、抽样、过滤都方便；
    - 更适合从 `/eval`、badcase、人工标注工具中逐步积累样本；
    - 比一个大 JSON 更适合做数据管道中的中间产物。

    `limit` 的作用不是正式训练必需，而是：
    - smoke test
    - 小样本试训
    - 先验证模板和依赖是否通
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no training rows found in {path}")
    return rows


def normalize_messages_row(row: dict[str, Any]) -> list[dict[str, str]]:
    """统一解析训练样本的消息格式。

    当前优先支持两种输入：
    1. 标准 `messages`：
       {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    2. 简化字段：
       {
         "system": "...",
         "question": "...",
         "context": "...",
         "answer": "..."
       }

    第二种格式的作用是：方便从当前 RAG 数据和 badcase 数据快速转一版训练集。

    这里的关键工程取舍是：
    - 训练脚本不应该强绑某一种上游数据源；
    - 但也不能无限兼容各种随意结构，否则后面会很难维护。

    所以当前支持两种最值钱的输入：
    1. 标准 `messages`
       适合人工精标或整理好的高质量数据；
    2. `system/question/context/answer`
       适合从现有项目资产快速导出的第一版训练数据。
    """
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if role and content:
                normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError(f"invalid row, empty messages after normalization: {row}")
        return normalized

    system = str(row.get("system", "")).strip()
    question = str(row.get("question", row.get("query", ""))).strip()
    context = str(row.get("context", "")).strip()
    answer = str(row.get("answer", row.get("assistant", ""))).strip()

    if not question or not answer:
        raise ValueError(f"invalid row, missing question/answer: {row}")

    # 这里把 context 拼进 user message，而不是单独做一个自定义 role，
    # 是因为大部分 instruct/chat 模型的原生模板都天然支持 system/user/assistant。
    # 如果引入额外 role，反而容易和模型原生 chat template 不兼容。
    user_content = question
    if context:
        user_content = f"已检索证据：\n{context}\n\n用户问题：{question}"

    normalized = []
    if system:
        normalized.append({"role": "system", "content": system})
    normalized.append({"role": "user", "content": user_content})
    normalized.append({"role": "assistant", "content": answer})
    return normalized


def fallback_chat_template(messages: list[dict[str, str]]) -> str:
    """在 tokenizer 不支持 `apply_chat_template` 时的保底拼接。

    当前不追求花哨模板，而是保证：
    - role 边界清晰
    - 文本可读
    - 训练时至少能保持基本对话结构

    为什么要有这个 fallback：
    - 某些本地模型目录没有完整 `chat_template`
    - 某些 tokenizer 虽然有 `apply_chat_template`，但模板定义可能不完整
    - 训练脚本不能因为模板小问题直接完全不可用
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def format_messages_for_training(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> str:
    """把消息列表格式化成训练文本。

    优先级：
    1. 如果 tokenizer 支持 `apply_chat_template`，优先复用模型原生模板；
    2. 否则回退到当前脚本自己的简化模板。

    这样做的原因：
    - 尽量保持和目标基础模型的对话格式一致；
    - 避免手工拼接模板和模型原生模板完全错位。

    这是本地生成模型微调里一个非常重要的点：

    训练格式如果和推理格式差异太大，常见后果是：
    1. 模型学到一套“训练时的说话方式”
    2. 部署推理时又喂另一套模板
    3. 最终表现为：
       - 拒答风格不稳定
       - 引用占位格式飘
       - 冲突说明不稳定

    所以这里优先遵守模型原生模板，是为了减少“模板分布漂移”。
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # 某些本地模型目录即使有这个方法，也可能因为模板不完整失败，
            # 所以这里保留一个静默回退。
            pass
    return fallback_chat_template(messages)


def save_run_manifest(cfg: TrainConfig, train_rows: int) -> None:
    """保存本次微调的关键配置快照。

    对生成模型微调来说，这一步尤其重要。

    因为后面你经常会碰到这种情况：
    - 有多个 adapter 版本
    - 都叫 `v1 / v2 / final / final2`
    - 但已经忘了到底哪个版本：
      - 用了多少样本
      - 多长上下文
      - 什么学习率
      - 哪个基础模型

    `run_manifest.json` 的价值，就是把“模型产物”和“训练上下文”绑在一起。
    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "train_rows": train_rows,
        "training_method": "qlora_sft",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> TrainConfig:
    """解析命令行参数。

    这里的参数设计思路是：
    - 先暴露最关键、最常调的参数
    - 不把所有训练框架底层细节都暴露出来
    - 避免第一版脚本变成“开关很多但没人敢改”的状态

    也就是说，这份脚本的目标不是覆盖所有训练可能性，
    而是给当前项目提供一条最稳的第一轮 `QLoRA + SFT` 落地路径。
    """
    parser = argparse.ArgumentParser(
        description="对本地生成式模型做第一轮 QLoRA/SFT 微调。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="基础生成模型目录或模型名，建议指向本地 `Qwen3-14B-Instruct`。",
    )
    parser.add_argument(
        "--train-path",
        required=True,
        help="训练集 JSONL 路径。推荐每行使用 `messages` 结构。",
    )
    parser.add_argument(
        "--dev-path",
        default=None,
        help="可选验证集路径。当前脚本预留参数，第一轮可以先不提供。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="LoRA/QLoRA 适配器输出目录。",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="单条样本最大序列长度。更大通常更吃显存。",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="单卡 batch size。生成模型 QLoRA 通常从 1 起步最稳。",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="梯度累计步数，用于在低显存下获得更大的有效 batch。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="QLoRA/SFT 学习率的稳妥起点。",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=2,
        help="训练轮数。第一轮建议先保守试训。",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="日志打印间隔。",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="checkpoint 保存间隔。",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank。越大参数越多、可学习容量越高，也越吃显存。",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA 缩放系数，通常与 `r` 一起成对调节。",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout。第一轮一般保持小值即可。",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="启用 4bit 量化加载，适合 QLoRA 训练。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--train-max-rows",
        type=int,
        default=None,
        help="只取前 N 行训练数据，常用于 smoke test。",
    )
    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        train_path=args.train_path,
        dev_path=args.dev_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
        train_max_rows=args.train_max_rows,
    )


def import_training_stack() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """延迟导入训练依赖。

    这样做的目的，是让当前脚本在“本机只看帮助或做语法校验”时不强依赖：
    - transformers
    - peft
    - trl
    - bitsandbytes
    - datasets

    真实训练时如果这些包没装，会给出明确报错提示。

    这也是为什么当前脚本能在 macOS 开发机上：
    - 先过 `py_compile`
    - 先看 `--help`
    - 先整理样本

    但真正训练仍建议在 Linux + CUDA 训练机上做。
    """
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "缺少 QLoRA/SFT 训练依赖，请先在 Linux + CUDA 环境安装："
            "`transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `accelerate`。"
        ) from exc

    return (
        Dataset,
        LoraConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        SFTConfig,
        SFTTrainer,
    )


def main() -> None:
    """训练主入口。

    主流程：
    1. 读取 JSONL 训练样本；
    2. 统一成 `messages` 结构；
    3. 用 tokenizer/chat template 格式化成训练文本；
    4. 以 QLoRA 方式加载基础模型；
    5. 用 `trl.SFTTrainer` 执行第一轮监督微调。

    这里用 `SFTTrainer` 而不是自己手写 Trainer 的原因：
    - 当前目标是稳定落地，不是做训练框架实验；
    - `trl` 对 chat-style SFT 更顺手；
    - 和 `peft + transformers` 的配合更成熟。
    """
    configure_logging()
    cfg = parse_args()
    set_seed(cfg.seed)

    logger.info("loading training rows from %s", cfg.train_path)
    train_rows = load_jsonl(cfg.train_path, limit=cfg.train_max_rows)
    save_run_manifest(cfg, train_rows=len(train_rows))

    Dataset, LoraConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, SFTConfig, SFTTrainer = (
        import_training_stack()
    )

    logger.info("loading tokenizer from %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        # 很多 causal LM 本身没有显式 pad_token。
        # 训练时如果不补齐 pad_token，容易在 batch padding 阶段出问题。
        # 这里把 eos 当 pad，是当前很多 SFT/QLoRA 脚本常见的保守处理。
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("formatting training rows into chat text")
    normalized_rows = []
    for row in train_rows:
        # 先统一成 messages，再格式化成最终训练文本。
        # 这样后面如果你要从 badcase、audit 或评测资产导样本，只要能转成 messages，
        # 就不需要改底层训练逻辑。
        messages = normalize_messages_row(row)
        text = format_messages_for_training(messages, tokenizer)
        normalized_rows.append({"text": text})

    train_dataset = Dataset.from_list(normalized_rows)

    quantization_config = None
    if cfg.load_in_4bit:
        # QLoRA 的关键原理：
        # 1. 基础模型以 4bit 量化方式加载，显著降低显存占用；
        # 2. 训练时不直接更新全部大模型权重；
        # 3. 只训练少量 LoRA adapter 参数。
        #
        # 这样 14B 模型也能在相对可控的 GPU 资源上完成第一轮领域对齐。
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("loading base model from %s", cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # 当前默认目标模块覆盖主流 Qwen 注意力投影层。
        # 如果后续换模型，再按该模型结构调整即可。
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 这里的训练参数组合，本质上是在做一个现实工程上的平衡：
    # - per_device_train_batch_size=1：减少单步显存压力
    # - gradient_accumulation_steps=8：把多个小步累加成更大的有效 batch
    # - bf16=True：在支持的 GPU 上通常更稳，也比 fp32 更省资源
    #
    # 这套默认值不是“理论最优”，而是对当前企业项目第一轮本地对齐更稳的起点。
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        bf16=True,
        max_seq_length=cfg.max_seq_length,
        report_to=[],
    )

    logger.info(
        "start QLoRA training: rows=%d batch_size=%d grad_acc=%d lr=%s epochs=%d",
        len(train_rows),
        cfg.per_device_train_batch_size,
        cfg.gradient_accumulation_steps,
        cfg.learning_rate,
        cfg.num_train_epochs,
    )

    # `SFTTrainer` 在这里承担的是：
    # - tokenization 后的数据喂入
    # - loss 计算
    # - checkpoint 保存
    # - 训练循环组织
    #
    # 当前我们故意不在这里叠更多复杂技巧，比如：
    # - packing
    # - 自定义 reward
    # - DPO / ORPO
    #
    # 因为这份脚本的目标是第一轮“把 grounded/refusal/conflict 对齐起来”，
    # 不是一步到位做最复杂的偏好优化。
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)

    logger.info("training finished, adapter saved to %s", cfg.output_dir)
    logger.info(
        "next step: deploy the base model + LoRA adapter in local OpenAI-compatible serving, then replay /eval and high-sensitive badcases"
    )


if __name__ == "__main__":
    main()
