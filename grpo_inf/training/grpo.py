from __future__ import annotations

import json
from typing import Any

from grpo_inf.rewards.reviewer_reward import reward_func
from grpo_inf.training.common import dry_run_summary, prepare_training_run
from grpo_inf.training.config import load_config


def _trl_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    return reward_func(completions, **kwargs)


def run_grpo(config_path: str, run_id: str | None = None, execute: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    run_id, paths = prepare_training_run(config, "grpo", run_id, f"train-grpo --config {config_path}")
    if not execute:
        summary = dry_run_summary(config, "grpo", run_id, paths)
        (paths["logs"] / "dry_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("json", data_files=config["data_path"], split="train")
    peft_config = LoraConfig(**config.get("lora", {})) if config.get("lora") else None
    grpo_args = dict(config["grpo"])
    grpo_args.setdefault("output_dir", str(paths["checkpoints"]))
    grpo_args.setdefault("bf16", bool(config.get("bf16", True)))
    grpo_args.setdefault("report_to", config.get("report_to", "none"))
    grpo_args.setdefault("chat_template_kwargs", {"enable_thinking": False})
    trainer = GRPOTrainer(
        model=config["model_id"],
        reward_funcs=_trl_reward,
        args=GRPOConfig(**grpo_args),
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(paths["adapter"]))
    return {"mode": "execute", "run_id": run_id, "adapter_dir": str(paths["adapter"])}
