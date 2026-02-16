# Chronos v3: RLVR Training Loop Design

**Date**: 2026-02-16
**Status**: Design document — grounded in Phase A/B1 data + literature review

---

## 1. The Setting

We have a sequential optimization MDP:
- **State**: PTX kernel features (25 scalar features from ParsedKernel)
- **Action**: one of 21 discrete actions (20 transforms + stop)
- **Reward**: hardware-measured cycle delta (SM clock(), 1-cycle std dev)
- **Episode**: 1-5 steps (mean 3.0 from greedy v2)
- **Kernels**: 64 gemm_tile variants (action space validated, BC baseline exists)

Data assets:
- 259 trajectory entries (195 transform + 64 stop) from greedy v2
- BC baseline: 59.1% RF accuracy, 73.4% first-action accuracy
- 20 validated transforms with conflict groups
- Greedy v2 ceiling: -28.1% mean, -54.3% best

---

## 2. Algorithm Choice: GRPO

### Why GRPO over PPO

| Criterion | PPO | GRPO | For Chronos |
|-----------|-----|------|-------------|
| Value network | Required (doubles params) | Not needed | GRPO wins — simpler |
| Advantage estimation | GAE with learned baseline | Group-relative normalization | GRPO wins — no critic to train |
| Stability | Mature, well-understood | Newer, active research | PPO safer but GRPO simpler |
| Group size | N/A | G=8-16 per prompt | 8 rollouts/kernel feasible |
| KL regularization | Reward penalty | Loss penalty | Similar |

### Why not evolutionary (AlphaEvolve)

AlphaEvolve uses LLMs as mutation engines with evolutionary selection. This requires large LLMs (Gemini Flash/Pro) and population-based search. Our setting has:
- Discrete, small action space (21 actions)
- Short episodes (1-5 steps)
- Fast evaluation (50ms)

These favor gradient-based RL over evolutionary methods. AlphaEvolve shines for unbounded code modification; we have a fixed action library.

### Why not a world model (CompilerDream)

CompilerDream learns a model of the LLVM compiler to avoid real compilation. For Chronos:
- Our "compiler" (ptxas) is a black box with chaotic behavior (see A5 findings)
- Reorder impact ranges from -21.7% to +30.4% — hard to model
- Hardware measurement is fast enough (50ms) for online RL
- CuAsmRL proved hardware-in-the-loop PPO works at this timescale

A world model could help later but adds complexity without clear benefit now.

### Closest prior work: CuAsmRL (CGO 2025)

CuAsmRL applies PPO to SASS instruction scheduling:
- Small CNN encoder, actor-critic
- Hardware-in-the-loop reward: `R = (T_prev - T_curr) / T_initial * 100`
- 15,000 episodes, <5 hours per kernel
- Up to 26% improvement (our greedy v2: 28.1% mean)
- They restrict actions to memory instruction reordering only

Our setting is structurally similar but at the PTX level with a broader action set.

---

## 3. Reward Function Design

### 3.1 Per-step reward: log-transformed cycle ratio

```
r_t = log(cycles_{t-1} / cycles_t)
```

**Why log-transform?**
- Converts multiplicative speedups to additive: total reward = sum of per-step rewards
- Symmetrizes distribution: -54.3% speedup → log(1/0.457) = +0.78, +30% regression → log(1/1.30) = -0.26
- Matches CompilerDream's finding that log-latency is the right target
- Reduces variance from extreme outliers

**Example (gemm_tile 4,6,8, best kernel):**

| Step | Action | Cycles | r_t (raw) | r_t (log) |
|------|--------|--------|-----------|-----------|
| 0→1 | maxnreg_128 | 1829→1103 | -0.397 | +0.506 |
| 1→2 | vec_ld | 1103→909 | -0.176 | +0.193 |
| 2→3 | vec_st | 909→841 | -0.075 | +0.078 |
| 3→4 | reorder_cp | 841→835 | -0.007 | +0.007 |
| | **Total** | 1829→835 | -0.543 | **+0.784** |

Sum of log rewards = 0.506 + 0.193 + 0.078 + 0.007 = 0.784 = log(1829/835). Additive decomposition holds exactly.

### 3.2 Stop action reward

Stop gets exactly 0 additional reward. The agent's incentive to stop is:
1. **Per-step cost**: small penalty c = -0.005 per step (discourages wasteful exploration)
2. **Risk aversion**: applying a bad transform gives negative reward; stopping avoids risk
3. **No positive stop reward** (prevents always-stop hacking, per Weng 2024)

```
r_stop = -c = -0.005
```

The per-step cost means the agent must believe the next transform will yield at least +0.005 log-improvement (~0.5% speedup) to justify continuing.

### 3.3 Full episode reward

```
R_episode = sum_t [r_t - c] for all steps including stop
         = log(cycles_baseline / cycles_final) - c * n_steps
```

This is outcome-equivalent (total speedup) but process-grounded (per-step signal).

### 3.4 Why not terminal-only reward?

With 1-5 step episodes, terminal-only reward is not catastrophically sparse. But we have ground-truth intermediate measurements — using them gives strictly more signal. The process-vs-outcome literature (2025) shows process rewards achieve similar results with 18x fewer training samples.

### 3.5 Reward normalization

Two-step normalization following REINFORCE++ (2025):

**Step 1**: Per-kernel baseline subtraction
```
A_i = R_i - mean(R_1...R_G)  for G rollouts of same kernel
```

**Step 2**: Global batch normalization
```
A_norm = (A_i - mean_batch(A)) / (std_batch(A) + epsilon)
```

This handles heterogeneous kernel difficulty (some kernels have -50% potential, others 0%).

For robustness with small group sizes (G=8), use MC-GRPO's median-centered baseline:
```
A_i = R_i - median(R_1...R_G)
```

---

## 4. Model Architecture

### 4.1 Start with MLP on features (not LLM on PTX)

Rationale:
- BC baseline achieves 59% accuracy with 25 features — signal exists
- 270M LLM on PTX text: failed at zero-shot in v2 (collapsed)
- "Limit of RLVR" paper (NeurIPS 2025): RLVR only surfaces existing capabilities
- A 270M LLM likely lacks sufficient PTX understanding for RLVR to amplify
- CuAsmRL uses a small CNN with PPO — works well for instruction scheduling
- Feature-based model is faster (no tokenization, no attention) → more rollouts per second

### 4.2 Architecture

```
Input: 25 scalar features (kernel state)
       + 21-dim action mask (available actions)
       + 21-dim action history (which transforms already applied, one-hot sum)

Hidden: MLP
  Linear(67, 128) → ReLU → Dropout(0.1)
  Linear(128, 128) → ReLU → Dropout(0.1)

Output: 21-dim logits (masked by action_mask before softmax)

Total parameters: ~20K
```

**Why include action history?** Non-monotone interactions mean the same kernel state with different applied transforms should lead to different decisions. Action history captures "maxnreg_128 already applied, so reorder_cp might now help."

### 4.3 Later: upgrade to LLM on PTX

Once the feature-based model establishes a baseline, we can:
1. Fine-tune CodeGemma 2B on PTX text with GRPO
2. Use the feature-based model as a teacher/baseline
3. Compare: does reading PTX add value beyond scalar features?

---

## 5. Training Loop

### 5.1 Overview

```
for epoch in range(N_EPOCHS):
    # 1. Sample batch of kernels
    kernel_batch = sample(all_kernels, batch_size=32)

    # 2. For each kernel, generate G rollouts
    for kernel in kernel_batch:
        for g in range(G):
            rollout = generate_rollout(kernel, policy)
            measure_cycles(rollout)  # hardware-in-the-loop

    # 3. Compute advantages (GRPO)
    advantages = grpo_advantage(rollouts)

    # 4. Policy gradient update
    loss = clipped_surrogate_loss(advantages)
    optimizer.step()
```

### 5.2 Rollout generation

```python
def generate_rollout(kernel, policy, max_steps=6):
    """Generate one episode for a kernel."""
    ptx = kernel.base_ptx
    applied = set()
    trajectory = []

    for step in range(max_steps):
        # Extract features from current PTX
        parsed = parse_kernel(ptx)
        features = extract_features(parsed)
        action_mask = get_available_actions(applied)
        action_history = encode_applied(applied)

        # Sample action from policy
        state = concat(features, action_mask, action_history)
        action = policy.sample(state, action_mask)

        if action == STOP:
            trajectory.append((state, STOP, -STEP_COST))
            break

        # Apply transform and measure
        new_ptx, changed = apply_transform(ptx, action)
        if not changed:
            trajectory.append((state, action, -STEP_COST))  # no-op penalty
            break

        cycles_before = measure(ptx, kernel)
        cycles_after = measure(new_ptx, kernel)
        reward = log(cycles_before / cycles_after) - STEP_COST

        trajectory.append((state, action, reward))
        ptx = new_ptx
        applied.add(action)

    return trajectory
```

### 5.3 GRPO advantage computation

```python
def grpo_advantage(rollouts_per_kernel):
    """Compute group-relative advantages.

    rollouts_per_kernel: dict[kernel_id -> list of G rollout rewards]
    """
    all_advantages = []

    for kernel_id, rollouts in rollouts_per_kernel.items():
        episode_rewards = [sum(r for _, _, r in traj) for traj in rollouts]

        # Per-kernel baseline (median for robustness)
        baseline = median(episode_rewards)

        for traj, R in zip(rollouts, episode_rewards):
            advantage = R - baseline
            # Same advantage for all steps in the trajectory (outcome-level)
            for state, action, reward in traj:
                all_advantages.append((state, action, advantage))

    # Global normalization
    advs = [a for _, _, a in all_advantages]
    mean_a, std_a = mean(advs), std(advs)
    normalized = [(s, a, (adv - mean_a) / (std_a + 1e-8))
                  for s, a, adv in all_advantages]

    return normalized
```

### 5.4 Policy gradient loss

```python
def compute_loss(policy, old_policy, advantages, beta=0.01, epsilon=0.2):
    """Clipped surrogate loss with KL penalty (GRPO)."""
    total_loss = 0

    for state, action, advantage in advantages:
        # Importance ratio
        log_prob = policy.log_prob(state, action)
        old_log_prob = old_policy.log_prob(state, action)
        ratio = exp(log_prob - old_log_prob)

        # Clipped surrogate
        surr1 = ratio * advantage
        surr2 = clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
        policy_loss = -min(surr1, surr2)

        # KL penalty against reference policy
        kl = old_log_prob - log_prob  # approximate KL
        kl_penalty = beta * kl

        total_loss += policy_loss + kl_penalty

    return total_loss / len(advantages)
```

### 5.5 Training schedule

| Phase | Epochs | What happens |
|-------|--------|-------------|
| **Warm-start (BC)** | 0-50 | Clone greedy v2 trajectories. Initialize policy weights. |
| **Exploration** | 50-200 | GRPO with high temperature (T=1.5). Agent explores beyond greedy. |
| **Exploitation** | 200-500 | Lower temperature (T=0.5). Agent refines discovered strategies. |
| **Evaluation** | Every 50 | Run full greedy-style eval on all 64 kernels with learned policy. Compare to greedy v2 (-28.1%). |

### 5.6 Compute budget

Per training step (32 kernels × 8 rollouts × ~3 steps/rollout):
- Rollout measurements: 32 × 8 × 3 × 50ms = **38.4 seconds**
- Policy forward/backward: ~0.1 seconds (20K param MLP)
- Total per step: ~40 seconds
- Steps per epoch: ~2 (64 kernels / 32 per batch)
- **Per epoch: ~80 seconds**
- **500 epochs: ~11 hours**

This is comparable to CuAsmRL's <5 hours per kernel. Our training covers all 64 kernels simultaneously.

---

## 6. What Can Go Wrong (and Mitigations)

### 6.1 Always-stop policy

**Risk**: Agent learns to always predict "stop" (0 reward, no risk)
**Mitigation**:
- Per-step cost means stop gives -0.005, not 0
- Curriculum: force 2+ steps in early training, introduce stop at epoch 50
- KL penalty against BC reference policy that rarely stops at step 0

### 6.2 Mode collapse to vec_ld + vec_st

**Risk**: Agent learns to always predict vec_ld, vec_st (works for 80%+ of kernels)
**Mitigation**:
- Per-kernel baseline subtraction removes easy kernels from gradient
- Temperature annealing preserves exploration
- Diverse kernel training (future: add Triton kernels)

### 6.3 Measurement noise exploitation

**Risk**: Agent finds actions that exploit cycle measurement noise
**Mitigation**:
- SM clock() has 1-cycle std dev — noise is negligible relative to transform effects
- Require >1% improvement (>4 cycles on small kernels) to count as positive
- Use median of 200 measurements (already implemented)

### 6.4 Overfitting to 64 kernels

**Risk**: Policy memorizes per-kernel strategies, fails on new kernels
**Mitigation**:
- Features are kernel-agnostic (instruction counts, ratios — not kernel identity)
- LOKO evaluation reveals generalization
- Plan: add diverse kernels (Triton, inductor) before final training

### 6.5 Non-monotone composition failures

**Risk**: Agent learns A→B but not that B→A is different
**Mitigation**:
- Action history in state representation captures ordering
- GRPO's multiple rollouts per kernel explore different orderings
- Log-additive reward correctly decomposes multi-step effects

---

## 7. Success Criteria

| Metric | Greedy v2 Baseline | Target | Meaning |
|--------|-------------------|--------|---------|
| Mean improvement | -28.1% | **> -28.1%** | Agent finds better sequences than greedy |
| First-action accuracy | 73.4% (BC) | **> 80%** | Agent predicts right first move |
| LOKO accuracy | 59.1% (RF) | **> 70%** | Policy generalizes across kernels |
| Novel sequences | 0 | **> 5** | Agent discovers sequences greedy missed |
| Training time | — | **< 24 hours** | Feasible on single L4 GPU |

The key test: does the RLVR agent discover transform sequences that the greedy search missed? Greedy is myopic (takes best single step). The agent should learn multi-step lookahead — e.g., apply a neutral transform that enables a larger improvement on the next step.

---

## 8. References

### Direct prior work
- **CuAsmRL** (CGO 2025): PPO on SASS scheduling, hardware-in-the-loop, up to 26% speedup. https://arxiv.org/abs/2501.08071
- **CompilerDream** (KDD 2025): DreamerV3 world model for LLVM pass ordering. https://arxiv.org/abs/2404.16077
- **Dr. Kernel** (2026): REINFORCE for Triton kernel generation, 8B/14B LLMs, beats GPT-5. https://arxiv.org/abs/2602.05885
- **Compiler-R1** (2025): Agentic compiler auto-tuning with RL, synergy graph. https://arxiv.org/abs/2506.15701
- **Pearl** (ICS 2025): PPO for compiler optimization sequences, 2.02x speedup. https://arxiv.org/abs/2506.01880

### GRPO algorithm
- **DeepSeek-R1** (2025): GRPO, G=16, beta=0.001, epsilon=10. https://arxiv.org/abs/2501.12948
- **DAPO** (ByteDance, 2025): GRPO fixes (clip-higher, dynamic sampling). https://arxiv.org/abs/2503.14476
- **MC-GRPO** (2025): Median-centered baseline for small groups. https://arxiv.org/abs/2601.22582
- **GRPO bias analysis** (2025): Group size vs advantage accuracy. https://arxiv.org/abs/2601.08521
- **REINFORCE++** (2025): Two-step normalization. https://arxiv.org/abs/2501.03262

### Reward design
- **Limit of RLVR** (NeurIPS 2025): RLVR surfaces existing capabilities, doesn't create new ones. https://limit-of-rlvr.github.io/
- **Process vs Outcome rewards** (2025): Process rewards need 18x fewer samples. https://arxiv.org/abs/2505.14069
- **GRPO is secretly PRM** (2025): GRPO implicitly does process reward. https://arxiv.org/abs/2509.21154
- **Reward hacking survey** (Weng, 2024): Failure modes. https://lilianweng.github.io/posts/2024-11-28-reward-hacking/

### Small model RL
- **Qwen-0.5B with GRPO** (2025): Works, beats O1-preview on ROUGE-L. Smallest confirmed GRPO success.
- **1-shot RLVR** (2025): Single training example elevates Qwen2.5-Math-1.5B from 36% to 73.6%. https://arxiv.org/abs/2504.20571
- **DeepSeek finding**: Distillation outperforms direct RL on small models.
