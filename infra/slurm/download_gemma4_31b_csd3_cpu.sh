#!/usr/bin/env bash
#SBATCH --account=mphil-dis-sl2-cpu
#SBATCH --partition=icelake
#SBATCH --job-name=gemma4-31b-download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
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
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MODEL_DIR="${MODEL_DIR:-$WORK_ROOT/models/gemma-4-31B-it}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/outputs/grpo-inf}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" \
  "$MODEL_DIR" "$OUTPUT_ROOT/slurm_logs" "$TMPDIR"

source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

echo "job_id=${SLURM_JOB_ID:-manual}"
echo "host=$(hostname)"
echo "model_dir=$MODEL_DIR"
echo "hf_home=$HF_HOME"
date -u
python -V
python - <<'PY'
import importlib.metadata as md
print("huggingface_hub", md.version("huggingface_hub"))
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
