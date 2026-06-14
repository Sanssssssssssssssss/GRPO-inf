from __future__ import annotations

import json
from typing import Any

from grpo_inf.rewards.reviewer_reward import reward_func
from grpo_inf.training.common import dry_run_summary, prepare_training_run
from grpo_inf.training.config import load_config


REPO_LOCAL_GRPO_KEYS = {"notes"}


def _trl_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    return reward_func(completions, **kwargs)


def _grpo_config_args(raw_args: dict[str, Any], config_cls: type[Any]) -> dict[str, Any]:
    valid_keys = set(getattr(config_cls, "__dataclass_fields__", {}))
    if not valid_keys:
        return {key: value for key, value in raw_args.items() if key not in REPO_LOCAL_GRPO_KEYS}
    return {
        key: value
        for key, value in raw_args.items()
        if key in valid_keys and key not in REPO_LOCAL_GRPO_KEYS
    }


def _set_tokenizer_eos_token_id(processing_class: Any, eos_token_id: int) -> None:
    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    old_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)
    if eos_token is None or eos_token == getattr(tokenizer, "unk_token", None):
        raise ValueError(f"Cannot resolve EOS token id {eos_token_id!r} in tokenizer")
    tokenizer.eos_token = eos_token
    if getattr(tokenizer, "eos_token_id", None) != eos_token_id:
        try:
            tokenizer.eos_token_id = eos_token_id
        except AttributeError:
            pass
    if getattr(tokenizer, "eos_token_id", None) != eos_token_id:
        raise ValueError(
            f"Failed to set tokenizer eos_token_id to {eos_token_id}; "
            f"current value is {getattr(tokenizer, 'eos_token_id', None)!r}"
        )
    print(
        f"[INFO] TRL tokenizer eos_token_id override for clipped/mask logic: "
        f"{old_eos_token_id!r} -> {eos_token_id!r} ({eos_token!r})",
        flush=True,
    )


def _sync_model_eos_token_id(model: Any, eos_token_id: int) -> None:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = getattr(base_model, "config", None)
    if config is not None:
        config.eos_token_id = eos_token_id
    generation_config = getattr(base_model, "generation_config", None)
    if generation_config is not None:
        generation_config.eos_token_id = eos_token_id


def run_grpo(config_path: str, run_id: str | None = None, execute: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    run_id, paths = prepare_training_run(config, "grpo", run_id, f"train-grpo --config {config_path}")
    if not execute:
        summary = dry_run_summary(config, "grpo", run_id, paths)
        (paths["logs"] / "dry_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer
    from trl.trainer.utils import create_model_from_path

    dataset = load_dataset("json", data_files=config["data_path"], split="train")
    grpo_args = _grpo_config_args(dict(config["grpo"]), GRPOConfig)
    grpo_args.setdefault("output_dir", str(paths["checkpoints"]))
    grpo_args.setdefault("bf16", bool(config.get("bf16", True)))
    grpo_args.setdefault("report_to", config.get("report_to", "none"))
    grpo_args.setdefault("chat_template_kwargs", {"enable_thinking": False})
    model: str | Any = config["model_id"]
    peft_config = LoraConfig(**config.get("lora", {})) if config.get("lora") else None
    init_adapter_path = str(config.get("init_adapter_path") or "").strip()
    if init_adapter_path:
        model_init_kwargs = dict(grpo_args.get("model_init_kwargs") or {})
        model_init_kwargs["device_map"] = None
        base_model = create_model_from_path(config["model_id"], **model_init_kwargs)
        model = PeftModel.from_pretrained(base_model, init_adapter_path, is_trainable=True)
        peft_config = None

    processing_class = AutoProcessor.from_pretrained(
        config["model_id"],
        truncation_side="left",
        padding_side="left",
        local_files_only=bool(config.get("local_files_only", False)),
    )
    trl_eos_token_id = config.get("trl_eos_token_id_for_mask")
    if trl_eos_token_id is not None:
        _set_tokenizer_eos_token_id(processing_class, int(trl_eos_token_id))
        if not isinstance(model, str):
            _sync_model_eos_token_id(model, int(trl_eos_token_id))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_trl_reward,
        args=GRPOConfig(**grpo_args),
        train_dataset=dataset,
        processing_class=processing_class,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(paths["adapter"]))
    return {"mode": "execute", "run_id": run_id, "adapter_dir": str(paths["adapter"])}
