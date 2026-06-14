from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from peft import LoraConfig, get_peft_model
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ClippableLinear,
    Gemma4ForConditionalGeneration,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration

from grpo_inf.training.config import load_config


def _tiny_gemma4_config(model_dir: str) -> Any:
    config = copy.deepcopy(AutoConfig.from_pretrained(model_dir, local_files_only=True))
    is_unified = getattr(config, "model_type", None) == "gemma4_unified"
    config.text_config.num_hidden_layers = 1
    config.text_config.layer_types = ["sliding_attention"]
    config.text_config.hidden_size = 32
    config.text_config.intermediate_size = 64
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.num_global_key_value_heads = 2
    config.text_config.head_dim = 8
    config.text_config.global_head_dim = 8
    config.text_config.vocab_size = 128
    config.text_config.vocab_size_per_layer_input = 128
    config.text_config.max_position_embeddings = 128
    config.text_config.num_kv_shared_layers = 0

    if is_unified:
        config.vision_config = None
        config.audio_config = None
    else:
        config.vision_config.num_hidden_layers = 1
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_attention_heads = 4
        config.vision_config.num_key_value_heads = 4
        config.vision_config.head_dim = 8
        config.vision_config.global_head_dim = 8
        config.vision_config.position_embedding_size = 128
        config.vision_config.max_position_embeddings = 128
        config.vision_config.default_output_length = 4
        config.vision_soft_tokens_per_image = 4
    for attr, value in {
        "image_token_id": 90,
        "audio_token_id": 91,
        "video_token_id": 92,
        "boi_token_id": 93,
        "eoi_token_id": 94,
        "boa_token_id": 95,
        "eoa_token_id": 96,
    }.items():
        setattr(config, attr, value)
    return config


def _gemma4_model_class(config: Any) -> type[Any]:
    if getattr(config, "model_type", None) == "gemma4_unified":
        return Gemma4UnifiedForConditionalGeneration
    return Gemma4ForConditionalGeneration


def validate(config_path: str, model_dir: str) -> dict[str, Any]:
    training_config = load_config(config_path)
    lora_args = dict(training_config.get("lora") or {})
    if not lora_args:
        return {"valid": True, "skipped": True, "reason": "no lora config"}
    target_modules = lora_args.get("target_modules")
    if not isinstance(target_modules, str):
        raise ValueError("Gemma4 LoRA target_modules must be a regex string to avoid vision/audio wrappers.")
    if lora_args.get("target_parameters"):
        raise ValueError("Gemma4 LoRA target_parameters is not compatible with this ZeRO-3 path.")

    tiny_config = _tiny_gemma4_config(model_dir)
    model = _gemma4_model_class(tiny_config)(tiny_config)
    matched = [name for name, module in model.named_modules() if re.fullmatch(target_modules, name)]
    wrapper_matches = [
        name for name, module in model.named_modules() if re.fullmatch(target_modules, name) and isinstance(module, Gemma4ClippableLinear)
    ]
    if not matched:
        raise ValueError(f"No modules matched target_modules regex: {target_modules}")
    if wrapper_matches:
        raise ValueError(f"Gemma4ClippableLinear modules matched unexpectedly: {wrapper_matches}")

    peft_config = LoraConfig(**lora_args)
    peft_model = get_peft_model(model, peft_config)
    targeted = list(getattr(peft_model.base_model, "targeted_module_names", []))
    if not targeted:
        raise ValueError("PEFT did not target any modules.")
    return {
        "valid": True,
        "config": str(Path(config_path)),
        "model_dir": model_dir,
        "model_type": getattr(tiny_config, "model_type", None),
        "model_class": type(model).__name__,
        "matched_count": len(matched),
        "matched_modules": matched,
        "targeted_count": len(targeted),
        "targeted_modules": targeted,
        "trainable_params": sum(param.numel() for param in peft_model.parameters() if param.requires_grad),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Gemma4 LoRA regex targets on a tiny CPU model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.model_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
