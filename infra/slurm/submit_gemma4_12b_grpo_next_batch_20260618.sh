#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/rds-d6/user/cx272/hpc-work}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK_ROOT/outputs/grpo-inf}"
BEST_G8_ADAPTER="$OUTPUT_ROOT/runs/gemma4_12b_grpo_slim50_g8_beta001_from_sftfast40_20260616_g/adapter"
SLIM100_ADAPTER="$OUTPUT_ROOT/runs/gemma4_12b_grpo_slim100_lr3e7_from_sftfast40_20260616_h_retry/adapter"
BEST_G4_ADAPTER="$OUTPUT_ROOT/runs/gemma4_12b_grpo_slim50_continue_best10_20260616_b/adapter"
SFT_FAST40_ADAPTER="$OUTPUT_ROOT/runs/gemma4_12b_sft_slim_continue_fast40_20260615_d/adapter"

submit_slim() {
  local run_id="$1"
  local config_path="$2"
  local adapter_path="$3"
  sbatch --export=ALL,RUN_ID="$run_id",CONFIG_PATH="$config_path",INIT_ADAPTER_PATH="$adapter_path",SKIP_LORA_PREFLIGHT=1 \
    infra/slurm/train_gemma4_12b_grpo_slim_4xa100.sh
}

submit_compact() {
  local run_id="$1"
  local config_path="$2"
  local adapter_path="$3"
  sbatch --export=ALL,RUN_ID="$run_id",CONFIG_PATH="$config_path",INIT_ADAPTER_PATH="$adapter_path",SKIP_LORA_PREFLIGHT=1 \
    infra/slurm/train_gemma4_12b_grpo_compact_4xa100.sh
}

# Main line: continue from best 50-step G=8 beta adapter.
submit_slim gemma4_12b_grpo_next01_g8_cons_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta001_lr2e7.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next02_g8_var_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta002_temp07.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next03_g8_b003_t075_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta003_temp075.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next04_g8_mid_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta001_temp065_lr15e8.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next05_g8_nokl_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta000_temp075.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next06_g8_longlow_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_150step_g8_beta001_lr1e7.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next07_g8_cap1280_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_80step_g8_beta002_temp08_cap1280.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next08_g8_top097_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_80step_g8_beta003_temp08_top097.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next09_g4_ctrl_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g4_beta002_temp07.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next10_g8_b005_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta005_temp07.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next11_g8_lowtemp_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta001_temp05.json "$BEST_G8_ADAPTER"
submit_slim gemma4_12b_grpo_next12_g8_norep_bestg8_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta002_temp07_norep.json "$BEST_G8_ADAPTER"

# Recovery line: train on full compact targets from the same best slim-GRPO adapter.
submit_compact gemma4_12b_grpo_next13_compact_b002_bestg8_20260618 configs/training/gemma4_12b_grpo_compact_30step_from_slim_g8_beta002.json "$BEST_G8_ADAPTER"
submit_compact gemma4_12b_grpo_next14_compact_b003_bestg8_20260618 configs/training/gemma4_12b_grpo_compact_30step_beta003_temp065.json "$BEST_G8_ADAPTER"
submit_compact gemma4_12b_grpo_next15_compact50_bestg8_20260618 configs/training/gemma4_12b_grpo_compact_50step_beta002_lr2e7.json "$BEST_G8_ADAPTER"
submit_compact gemma4_12b_grpo_next16_compact_ctrl_bestg8_20260618 configs/training/gemma4_12b_grpo_compact_30step_beta001_temp05.json "$BEST_G8_ADAPTER"

# Starting-point controls.
submit_slim gemma4_12b_grpo_next17_g8_cons_slim100_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta001_lr2e7.json "$SLIM100_ADAPTER"
submit_slim gemma4_12b_grpo_next18_g8_var_slim100_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta002_temp07.json "$SLIM100_ADAPTER"
submit_slim gemma4_12b_grpo_next19_g8_b003_bestg4_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta003_temp075.json "$BEST_G4_ADAPTER"
submit_slim gemma4_12b_grpo_next20_g8_var_sftfast40_20260618 configs/training/gemma4_12b_grpo_slim_100step_g8_beta002_temp07.json "$SFT_FAST40_ADAPTER"
