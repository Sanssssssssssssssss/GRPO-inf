#!/usr/bin/env bash
#SBATCH --account=mphil-dis-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --job-name=gemma4-31b-probe4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=440G
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
nvidia-smi

hf download google/gemma-4-31B-it \
  --local-dir "$MODEL_DIR" \
  --max-workers "${HF_MAX_WORKERS:-8}"

RUN_ID="gemma4_31b_minimal_probe4_${SLURM_JOB_ID:-manual}"
accelerate launch \
  --config_file infra/slurm/accelerate_zero3_4xa100.yaml \
  grpo_inf/cli.py train-grpo \
  --config configs/training/gemma4_31b_grpo_probe_minimal.json \
  --run-id "$RUN_ID" \
  --execute
