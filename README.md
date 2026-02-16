# Chronos v3: RLVR Training for PTX Transform Selection

GRPO-based reinforcement learning from verifiable rewards (RLVR) for optimizing GPU kernel performance through PTX-level transform selection.

## What This Does

Given a GEMM tile kernel, the agent selects a sequence of PTX transforms (vectorize loads, cache hints, register budgets, instruction reorder, etc.) to minimize hardware cycle count. The reward signal comes from real SM clock() measurements on NVIDIA GPUs.

**Results so far:**
- Greedy v2 baseline: -28.1% mean cycle reduction across 64 kernels
- BC warm-start: 64.5% action prediction accuracy (MLP), 59.1% (Random Forest)
- GRPO training target: beat greedy by discovering multi-step lookahead strategies

## Requirements

- Python 3.10+
- PyTorch 1.13+
- NumPy
- **For hardware training**: NVIDIA GPU with CUDA (L4/T4 minimum), CuPy

BC warm-start and trajectory analysis work on CPU only.

```bash
pip install torch numpy
# For hardware-in-the-loop training:
pip install cupy-cuda12x
```

## Quick Start

```bash
# BC warm-start (CPU, ~2 seconds)
python train_rlvr.py --bc-only

# Quick test (2 kernels, no GPU)
python train_rlvr.py --quick --no-hardware

# Full GRPO training (requires NVIDIA GPU)
python train_rlvr.py

# Evaluate a checkpoint
python train_rlvr.py --eval --checkpoint results/rlvr/checkpoint_latest.pt
```

## Algorithm

**GRPO (Group Relative Policy Optimization)** — from DeepSeek-R1, adapted for compiler optimization:

1. **BC warm-start** (epochs 0-50): Clone greedy v2 trajectories with cross-entropy loss
2. **Exploration** (epochs 50-200): GRPO with temperature 1.5
3. **Exploitation** (epochs 200-500): GRPO with temperature 0.5

Key design decisions:
- **Reward**: `log(cycles_before / cycles_after) - 0.005` per step (log-additive decomposition)
- **Advantage**: MC-GRPO median baseline per kernel, global z-normalization
- **Loss**: Clipped surrogate (epsilon=0.2) + KL penalty against reference (beta=0.01)
- **No value network** — GRPO uses group-relative normalization instead

See [docs/TRAINING_LOOP_DESIGN.md](docs/TRAINING_LOOP_DESIGN.md) for full design with references.

## Action Space

21 discrete actions (20 transforms + stop):

| Category | Actions |
|----------|---------|
| Vectorize | vec_ld, vec_st, split_ld |
| Cache hints (load) | cache_cs, cache_cg, cache_ca, cache_cv |
| Cache hints (store) | st_cache_cs, st_cache_wt, st_cache_wb |
| Register budget | maxnreg_32, maxnreg_64, maxnreg_128, maxnreg_255 |
| Reorder | reorder_cp, reorder_il, reorder_lf, reorder_sl |
| Prefetch | prefetch_L1, prefetch_L2 |
| Terminal | stop |

Conflict groups prevent incompatible combinations (e.g., only one cache hint policy at a time).

## GPU Compatibility

This system requires **NVIDIA GPUs** for hardware-in-the-loop training:
- PTX is NVIDIA's intermediate representation
- `ptxas` is NVIDIA's PTX assembler
- SM clock() is an NVIDIA hardware register
- CuPy requires CUDA

AMD GPUs (MI300X, etc.) **cannot** run the hardware measurement loop. BC warm-start and analysis scripts work on any hardware.

## Data

All experimental data is in `data/`:
- **259 trajectory entries** from greedy v2 search across 64 gemm_tile kernels
- **64 kernels**: gemm_tile(m,n,k) for m,n,k in {2,4,6,8}
- Mean 3.0 steps per kernel, max 5 steps
- Top features: store_ratio, mem_ratio, maxnreg

## References

- [CuAsmRL](https://arxiv.org/abs/2501.08071) — PPO on SASS scheduling (CGO 2025)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO algorithm
- [MC-GRPO](https://arxiv.org/abs/2601.22582) — Median-centered baseline
- [Dr. Kernel](https://arxiv.org/abs/2602.05885) — REINFORCE for Triton kernels
- [CompilerDream](https://arxiv.org/abs/2404.16077) — World model for compiler optimization

Full reference list in [docs/TRAINING_LOOP_DESIGN.md](docs/TRAINING_LOOP_DESIGN.md).
