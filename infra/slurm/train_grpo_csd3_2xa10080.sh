#!/usr/bin/env bash
# EDIT THESE FOR CSD3 BEFORE SUBMITTING:
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=ampere
#SBATCH --job-name=grpo-inf-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm_logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p outputs/slurm_logs

: "${DATA_ROOT:?Set DATA_ROOT to the extracted invoice_reviewer_grpo_dataset_v0 directory on CSD3}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export RUN_ID="${RUN_ID:-gemma4_31b_grpo_${SLURM_JOB_ID:-manual}}"

echo "Run ID: ${RUN_ID}"
echo "DATA_ROOT: ${DATA_ROOT}"
date -u
nvidia-smi || true
python -V

accelerate launch \
  --config_file infra/slurm/accelerate_zero3_2xa100.yaml \
  grpo_inf/cli.py train-grpo \
  --config configs/training/gemma4_31b_grpo.json \
  --run-id "${RUN_ID}" \
  --execute
