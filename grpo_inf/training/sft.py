from __future__ import annotations

import json
from typing import Any

from grpo_inf.training.common import dry_run_summary, prepare_training_run
from grpo_inf.training.config import load_config


def run_sft(config_path: str, run_id: str | None = None, execute: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    run_id, paths = prepare_training_run(config, "sft", run_id, f"train-sft --config {config_path}")
    if not execute:
        summary = dry_run_summary(config, "sft", run_id, paths)
        (paths["logs"] / "dry_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.utils import create_model_from_path

    sft_config = config["sft"]
    dataset = load_dataset("json", data_files=config["data_path"], split="train")
    eval_dataset = None
    if config.get("eval_data_path"):
        eval_dataset = load_dataset("json", data_files=config["eval_data_path"], split="train")
    args = SFTConfig(
        output_dir=str(paths["checkpoints"]),
        bf16=bool(config.get("bf16", True)),
        report_to=config.get("report_to", "none"),
        **sft_config,
    )
    model: str | Any = config["model_id"]
    peft_config = LoraConfig(**config.get("lora", {})) if config.get("lora") else None
    init_adapter_path = str(config.get("init_adapter_path") or "").strip()
    if init_adapter_path:
        model_init_kwargs = dict(sft_config.get("model_init_kwargs") or {})
        model_init_kwargs["device_map"] = None
        base_model = create_model_from_path(config["model_id"], **model_init_kwargs)
        model = PeftModel.from_pretrained(base_model, init_adapter_path, is_trainable=True)
        peft_config = None
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(paths["adapter"]))
    return {"mode": "execute", "run_id": run_id, "adapter_dir": str(paths["adapter"])}
