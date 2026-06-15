from __future__ import annotations

from pathlib import Path


def test_slurm_scripts_are_execute_gated() -> None:
    for name in ("train_sft_csd3_2xa10080.sh", "train_grpo_csd3_2xa10080.sh"):
        text = Path("infra/slurm", name).read_text(encoding="utf-8")
        assert "#SBATCH --account=YOUR_ACCOUNT" in text
        assert "DATA_ROOT" in text
        assert "--execute" in text
        assert "outputs/slurm_logs" in text
        assert "accelerate launch" in text


def test_accelerate_config_targets_two_processes() -> None:
    text = Path("infra/slurm/accelerate_zero3_2xa100.yaml").read_text(encoding="utf-8")
    assert "distributed_type: DEEPSPEED" in text
    assert "num_processes: 2" in text
    assert "zero_stage: 3" in text


def test_accelerate_config_4xa100_uses_auto_gradient_clipping() -> None:
    text = Path("infra/slurm/accelerate_zero3_4xa100.yaml").read_text(encoding="utf-8")
    assert "num_processes: 4" in text
    assert "gradient_clipping: auto" in text
