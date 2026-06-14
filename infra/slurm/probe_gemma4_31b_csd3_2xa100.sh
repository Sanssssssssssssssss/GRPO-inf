#!/usr/bin/env bash
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --job-name=gemma4-31b-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=/rds-d6/user/cx272/hpc-work/outputs/grpo-inf/slurm_logs/%x-%j.out

set -euo pipefail

module purge >/dev/null 2>&1 || true
module load python/3.11.0-icl

export WORK_ROOT="${WORK_ROOT:-/rds-d6/user/cx272/hpc-work}"
export PROJECT_ROOT="${PROJECT_ROOT:-/home/cx272/final_project/GRPO-inf}"
export VENV="${VENV:-$WORK_ROOT/envs/grpo-inf-py311}"
export HF_HOME="${HF_HOME:-$WORK_ROOT/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/models/gemma-4-31B-it}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/outputs/grpo-inf}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" "$MODEL_DIR" "$OUTPUT_ROOT/slurm_logs" "$TMPDIR"

source "$VENV/bin/activate"
NVIDIA_LIB_PATHS="$(find "$VENV/lib" "$VENV/lib64" -path '*/site-packages/nvidia/*/lib' -type d 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT_ROOT"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "model_dir=$MODEL_DIR"
echo "output_root=$OUTPUT_ROOT"
echo "nvidia_lib_paths=$NVIDIA_LIB_PATHS"
date -u
python -V
python -m pip --version
nvidia-smi

python - <<'PY'
import importlib.metadata as md
for name in ("torch", "transformers", "trl", "accelerate", "datasets", "peft", "bitsandbytes", "deepspeed", "huggingface_hub"):
    print(f"{name}=={md.version(name)}")
PY

hf download google/gemma-4-31B-it \
  --local-dir "$MODEL_DIR" \
  --max-workers "${HF_MAX_WORKERS:-8}"

python - <<'PY'
from pathlib import Path
import os
required = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]
root = Path(os.environ["MODEL_DIR"])
missing = [name for name in required if not (root / name).exists()]
if missing:
    raise SystemExit(f"missing model files: {missing}")
for name in required:
    path = root / name
    print(f"{name}\t{path.stat().st_size}")
PY

du -sh "$MODEL_DIR" "$HF_HOME" || true

python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(idx)
    print(f"gpu{idx}", props.name, props.total_memory)
PY

python - <<'PY'
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_dir = os.environ["MODEL_DIR"]
processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype="auto",
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Reply with one short sentence confirming the model loaded."},
]
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=32)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print("short_generation", response.strip())
for idx in range(torch.cuda.device_count()):
    print(f"gpu{idx}_max_memory_allocated", torch.cuda.max_memory_allocated(idx))
PY

RUN_ID="gemma4_31b_minimal_probe_${SLURM_JOB_ID:-manual}"
accelerate launch \
  --config_file infra/slurm/accelerate_zero3_2xa100.yaml \
  grpo_inf/cli.py train-grpo \
  --config configs/training/gemma4_31b_grpo_probe_minimal.json \
  --run-id "$RUN_ID" \
  --execute

if [ "${RUN_REDUCED_PROBE:-0}" = "1" ]; then
  RUN_ID="gemma4_31b_reduced_probe_${SLURM_JOB_ID:-manual}"
  accelerate launch \
    --config_file infra/slurm/accelerate_zero3_2xa100.yaml \
    grpo_inf/cli.py train-grpo \
    --config configs/training/gemma4_31b_grpo_probe_reduced.json \
    --run-id "$RUN_ID" \
    --execute
fi
