# Project Chronos v3: Problem Definition

**Date**: 2026-02-16
**Status**: Phase A complete, Phase B1 complete (action space expansion)

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [Why This Problem Exists](#2-why-this-problem-exists)
3. [The Chronos Journey So Far](#3-the-chronos-journey-so-far)
4. [Analysis: The Fundamental Constraints (GEB/RL Thread)](#4-analysis-the-fundamental-constraints-gebrl-thread)
5. [Analysis: The Engineering Reality (Aaditya Conversation)](#5-analysis-the-engineering-reality-aaditya-conversation)
6. [The Sub-Problems](#6-the-sub-problems)
7. [What We Know From Data](#7-what-we-know-from-data)
8. [What We Don't Know](#8-what-we-dont-know)
9. [Open Questions](#9-open-questions)
10. [Research TODO](#10-research-todo)
11. [Phase A Research Results](#11-phase-a-research-results-2026-02-16)
12. [Phase A Revision: Reorder IS Performance-Relevant](#12-phase-a-revision-reorder-is-performance-relevant-2026-02-16-session-2)
13. [Phase B1 Results: Expanded Action Space](#13-phase-b1-results-expanded-action-space-2026-02-16)

---

## 1. The Problem

### One-sentence version

Build an agent that learns to optimize GPU kernel code by proposing modifications, replacing the millions of lines of hand-written compiler heuristics maintained by human engineers.

### Full statement

GPU kernel optimization is currently done by:
- Compiler heuristics: ptxas, LLVM, XLA, Triton — millions of lines of rules written by human engineers over decades
- Human experts: kernel engineers who read assembly, profile bottlenecks, and rewrite code
- Autotuning: brute-force search over configuration spaces (tile sizes, block dims)

None of these scale. Compilers apply fixed rules that miss non-obvious optimizations. Human experts are rare and expensive. Autotuning only searches configuration spaces, not code transformations.

The goal is an agent where:
- **Input**: a PTX kernel (any source — hand-written, Triton, inductor, CUTLASS)
- **Action**: a modification to the PTX that makes it faster
- **Validation**: compile + correctness check + hardware measurement
- **Learning**: the agent improves over time from its own experience

The actions are not limited to a fixed menu. They range from small (cache hints, register budgets) to large (vectorization, shared memory promotion, TMA replacement, algorithmic rewrites). The agent should eventually be capable of proposing modifications that no human has thought of — the way AlphaGo played moves that no human Go player had considered.

### What this is NOT

- Not a compiler pass (fixed rules applied uniformly)
- Not an autotuner (searching a predefined configuration space)
- Not a code generator (writing kernels from scratch)
- Not a classifier (picking the best option from a fixed menu)

It is a **learned optimization agent** that reads code, understands what it does and where it's slow, proposes targeted modifications, and learns from hardware feedback.

### Why this matters

Aaditya's framing: "There is no single heuristic that will let you optimize all programs by applying the heuristic recursively. Which is why pytorch/xla have millions of lines of optimizations hardcoded, not considering LLVM on top."

Those millions of lines are the accumulated knowledge of hundreds of engineers over decades. If an agent can learn even a fraction of that knowledge from experience, it compounds — every kernel it optimizes teaches it something about all future kernels. Human engineers don't transfer knowledge at that rate.

---

## 2. Why This Problem Exists

### The compiler gap

Modern GPU compilers (ptxas, Triton, XLA) are good at:
- Register allocation
- Basic instruction scheduling
- Standard optimizations (dead code elimination, constant folding, strength reduction)

They are bad at:
- Cross-kernel optimization (fusion, inter-kernel communication)
- Hardware-specific tricks (TMA, warp specialization, async copy pipelining)
- Workload-aware decisions (choosing tile sizes based on actual input shapes)
- Non-local trade-offs (accepting worse register pressure for better memory throughput)

The gap between what compilers produce and what expert humans write is 20-300% depending on the kernel type and hardware.

### The heuristic maintenance burden

Every new GPU architecture (Hopper → Blackwell → next) changes:
- Cache hierarchy and sizes
- Instruction latencies and throughputs
- New instructions (TMA, DPX, FP8)
- Occupancy characteristics
- Memory bandwidth and latency

Every one of these changes invalidates some subset of existing compiler heuristics. Engineers must update, test, and validate these heuristics for each new architecture. This is a permanent, growing maintenance cost.

An agent that learns from hardware feedback would automatically adapt to new architectures — run the same kernels, measure the new hardware, update the model. No manual heuristic rewriting.

### The search space

For a 131-instruction kernel (gemm_tile 4x4):
- Valid instruction orderings: ~10^146 (measured, see Chronos v1 data)
- PTX-level transforms (4 original): 12 configurations
- PTX-level transforms (20 validated, B1): hundreds of configurations
- Code-level modifications (rewrites, algorithm changes): effectively unbounded

No search algorithm can enumerate this space. The agent must learn to navigate it intelligently — proposing modifications that are likely to help, based on what it has learned from previous optimizations.

---

## 3. The Chronos Journey So Far

### v1: Instruction Scheduling (Phases 0-3)

**What**: MuZero-based scheduler for PTX instruction ordering. GNN policy on instruction DAGs, MCTS search, hardware evaluation via SM clock() cycle counter. 160 tests.

**Result**: Works — but hardware validation revealed a ceiling:
- gemm_tile 4x4 (131 instructions): 5.5% total optimization headroom (38 cycles on a 700-cycle kernel)
- Pipeline model: Pearson=0.958 but Spearman=0.462 (direction correct, ranking broken)
- BC model on full data: Spearman=-0.175 (worse than random)

**Why it failed**: Instruction reordering can't overcome architectural bottlenecks. A single cache miss costs ~500 cycles. The best possible instruction schedule saves 38 cycles. The ceiling is structural.

### v2: Transform-Based Optimization (Phases 6-8c)

**The pivot**: Instead of reordering instructions within a fixed program, transform the program itself. Started with 4 transforms (register_budget, cache_hints, reorder, vectorize_loads), expanded to 20 in B1.

**What worked**:

| Component | Result |
|-----------|--------|
| Transform library (4 → 20 transforms) | Up to -54.3% speedup (B1) |
| Greedy search v1 (4 transforms) | -20.1% mean across 64 kernels |
| Greedy search v2 (20 transforms) | **-28.1% mean** across 64 kernels |
| RF on 11 scalar features | +11.2%, 77% recovery, instant |
| SM clock() cycle counter | 1-cycle std dev, 38-sigma significance |
| Triton kernel scaling | Transforms generalize to compiler output |

**What failed**:

| Component | Result | Root Cause |
|-----------|--------|------------|
| GNN behavior cloning | +5.6%, collapsed to always-vectorize | 64 homogeneous kernels, 50% majority class |
| GNN on diverse data | +4.7%, vectorize/stop only | 76 kernels still too few, 64:12 imbalance |
| LLM 270M predictor | +5.0%, collapsed | Zero-shot prompting, model too small for reasoning |
| LLM with full PTX | +5.0% + 6 parse failures | PTX too verbose, few-shot bias |

**Current ceiling**: -28.1% mean with greedy search v2 over 20 transforms (up from -20.1% with 4 transforms).

### The Gap

Current state → Target state:
- **Actions**: 20 validated transforms (B1) → unbounded code modifications
- **Kernels**: 76 (64 gemm_tile + 12 Triton) → thousands of diverse real-world kernels
- **Learning**: classify-first-transform → multi-step optimization agent
- **Model**: RF on scalar features / collapsed GNN / collapsed LLM → agent that reads code and proposes targeted modifications

---

## 4. Analysis: The Fundamental Constraints (GEB/RL Thread)

A sustained analysis of the theoretical foundations through the lens of Gödel, Escher, Bach and modern RL. Key insights relevant to Chronos:

### 4.1 The Strange Loop of Learning to Optimize

The core challenge is circular:
- To optimize code well, the agent needs to **understand** code (representations)
- To understand code, the agent needs to have **seen many optimizations** (data)
- To collect optimization data, the agent needs to **explore** the transform space (actions)
- To explore well, the agent needs to **understand** which transforms might help (back to representations)

This is not a pipeline. It's a bootstrap problem. Every component depends on every other component already working.

### 4.2 Credit Assignment

v1 hit this directly. A scalar reward (cycle count) after 131 instruction placement decisions. The signal dilutes through the chain.

v2 partially addresses this by raising the abstraction level. Each transform is a macro-action with immediate, measurable effect. Apply vectorize → measure → know if it helped. The credit is per-transform, not per-instruction. This is like Options in Sutton's OaK architecture — temporal abstraction that reduces the credit assignment chain.

But for multi-step transform composition (the greedy search), credit assignment returns. If the sequence maxnreg_128 → vectorize → cache_cg gives -40%, how much credit goes to each step? The non-monotone interactions (cache_cg hurts alone but helps after vectorize) mean individual contributions can't be decomposed additively.

### 4.3 The Bitter Lesson Applied to Compilers

Sutton's bitter lesson: general methods + compute beat hand-crafted knowledge.

Compilers are the purest example of hand-crafted knowledge in computing. Millions of lines of human-written rules, each encoding a specific optimization pattern for a specific hardware scenario. This is exactly what the bitter lesson predicts will be replaced by learning.

But the bitter lesson requires SCALE — massive compute and data. AlphaEvolve needed Gemini Flash + Pro (hundreds of billions of parameters) to achieve 23% kernel speedup. The question: can a smaller model (200M-7B) learn enough from hardware feedback to be useful?

### 4.4 FC-STOMP Maps to Chronos

Sutton's FC-STOMP loop (Feature Construction → SubTask → Option → Model → Planning) has a direct mapping:

| FC-STOMP Stage | Chronos Equivalent |
|----------------|-------------------|
| Feature Construction | Understanding PTX structure and bottlenecks |
| SubTask | "Fix this bottleneck" (e.g., "reduce memory transactions") |
| Option | A learned optimization strategy (a transform or rewrite) |
| Model | Predicting what ptxas will do with modified PTX |
| Planning | Composing transforms into multi-step optimization sequences |

The two unsolved problems in FC-STOMP are also the two unsolved problems in Chronos:
1. **Catastrophic forgetting**: Can the agent learn new optimization strategies without losing old ones?
2. **The new-term problem**: How does the agent discover genuinely new optimization strategies (actions not in any fixed library)?

### 4.5 RLVR: The Verifier Advantage

Kernel optimization is a clean RLVR setting:
- The verifier is hardware (compile → correctness check → measure)
- The reward is ground truth (speedup ratio, not estimated, not approximated)
- Each action gets immediate feedback (no delayed reward)

This is structurally better than most RL domains. In chess, reward is delayed 50 moves. In robotics, reward is noisy and simulation-dependent. In kernel optimization, each modification gets a direct, hardware-measured verdict within 50ms.

The open question: does the RLVR structure translate into efficient learning for a small model, or does the model need to be large enough to "understand" code before RL can teach it to optimize?

### 4.6 The Counterfactual Credit Insight

From the theoretical analysis: instead of asking "which transform is best?" (classification with weak signal), ask "HOW does each transform change the kernel's execution?" (regression with richer signal).

For Chronos: instead of predicting the label "vectorize", predict the EFFECT — "merging 34 scalar loads into 9 vector loads reduces memory transactions by 73%." The effect is deterministic at the PTX level (before ptxas). The hardware impact depends on ptxas + execution, but the PTX-level effect is computable.

This decomposes the problem into:
1. What does the transform do to the code? (deterministic, static analysis)
2. What does ptxas do with the modified code? (black box, must measure)
3. How fast does the modified code run? (hardware, must measure)

Step 1 is free. Steps 2-3 cost 50ms. By exploiting step 1, the agent can filter out transforms that don't change anything meaningful BEFORE spending 50ms on evaluation.

---

## 5. Analysis: The Engineering Reality (Aaditya Conversation)

Key points from the 13-15 Feb 2026 WhatsApp exchange, deconstructed:

### 5.1 The Action Space Must Be Bigger

> "An action is MUCH bigger than a single reorder for example. An action looks like taking the gemm code out and replacing it with TMA for instance."

Current transforms operate at the directive/instruction level (add `.maxnreg`, change cache hints, merge loads). These are compiler-flag-level optimizations. The real gains come from algorithmic-level changes — restructuring how the kernel uses hardware.

Examples of "big actions" that human engineers do:
- Replace scalar global loads with TMA bulk copies
- Add shared memory tiling with double/triple buffering
- Insert software pipelining (overlap load of next tile with compute of current tile)
- Change from warp-uniform to warp-specialized execution
- Replace FP32 accumulation with FP16 accumulation + FP32 reduction
- Fuse multiple kernel launches into a single kernel

Each of these is a research-level code transformation — not a pattern match and replace but a structural rewrite that changes the algorithm while preserving semantics.

### 5.2 The Timetabling Analogy

Aaditya drew a parallel to timetabling:
- Both have combinatorial action spaces (~10^100+)
- Both have constraints that make most actions invalid
- Both have non-local credit assignment (changing one thing affects everything connected)
- Timetabling is "mostly solved since the 80s" using simulated annealing / evolutionary algorithms

Timetabling works because:
1. **Structured actions**: swap two time slots (not arbitrary mutations)
2. **Fast constraint checking**: is this valid? (nanoseconds)
3. **Graph-guided proposals**: fix the WORST constraint violation first
4. **Simple search works**: SA/evolutionary crushes it in minutes

Chronos equivalent:
1. Structured actions: parameterized transforms (partially built)
2. Fast validation: PTX parse + ptxas compile (~10ms, NOT 50ms for full measurement)
3. Profile-guided proposals: fix the bottleneck first (NOT built)
4. SA/evolutionary already implemented (Phase 7b/c) but with blind proposals

What's missing: #3. The proposals are blind — try all transforms, measure everything. Timetabling succeeds because proposals target the worst violation. Chronos needs proposals that target the actual bottleneck.

### 5.3 The Reward Noise Debate

Jayant's original concern was about measurement noise (~5% from thermal, jitter, OS scheduling). This has been **empirically resolved**: SM clock() cycle counter gives 1-cycle std dev with 38-sigma significance on small kernels, and 8-47 cycle std dev on large kernels (still <0.25% relative noise). The real constraint is the 50ms compile+measure cycle cost (50ms × 100K evaluations = 83 minutes — feasible).

### 5.4 The Engineering vs Research Tension

> "Training the model that does this is equal parts engineering as it is research."

The research questions:
- What model architecture?
- What training signal (RL, evolutionary, BC)?
- What action representation (discrete, parameterized, diffs)?
- How to grow the action space over time?
- How to transfer across kernels and hardware?

The engineering requirements:
- PTX extraction from diverse sources (PyTorch, CUTLASS, Triton, raw CUDA)
- Fast validation pipeline (parse → compile → correctness → measure)
- Training data collection at scale (landscape sweeps on diverse kernels)
- Model training infrastructure (fine-tuning on optimization data)
- Evaluation framework (fair comparison of approaches)

The engineering must come first. Without the data pipeline, no research direction can be evaluated.

### 5.5 Where Jayant and Aaditya Converge

Both agree on:
1. The goal: agent that proposes code modifications, validated on hardware
2. Start constrained: discrete set of known-valid transforms
3. Grow over time: expand action space as the system proves it can handle it
4. LLM as the backbone: a model that reads PTX and outputs modifications
5. Hardware as ground truth: no simulators, no approximations

Both are unsure about:
1. Model size: 200M enough? Need 7B? Need Gemini-scale?
2. Model-free vs model-based: does a world model help or is hardware feedback enough?
3. Timeline: months? years?
4. Generalization: same model across hardware? per-architecture fine-tuning?

---

## 6. The Sub-Problems

Breaking the problem into its component parts:

### 6.1 Code Understanding

**Problem**: The agent must "read" PTX and extract useful information — what the code computes, what hardware resources it uses, where the bottlenecks are.

**Current state**: GNN on instruction DAG (failed — collapsed on homogeneous data), RF on 11 scalar features (works but limited), LLM reading PTX as text (failed — 270M too small).

**What makes this hard**:
- PTX is verbose. A simple kernel has 100+ lines of boilerplate (register declarations, parameter loads). The optimization-relevant information is a small fraction.
- PTX is an intermediate representation, not the final ISA. What matters is SASS (what the GPU executes), but SASS is not directly visible before compilation.
- The same "understanding" must work across kernel types (matmul, reduction, attention, elementwise, scatter/gather) that have completely different structures.

**Open sub-question**: What representation of PTX captures the information the agent needs? Raw text? Instruction DAG? Abstract features? Some combination?

### 6.2 Bottleneck Diagnosis

**Problem**: Given a kernel, identify WHY it's slow — memory-bound? compute-bound? latency-bound? occupancy-limited? Which specific instructions or access patterns are the bottleneck?

**Current state**: Not addressed in Chronos v1 or v2. Transforms are applied blindly (try everything, measure).

**What makes this hard**:
- Requires profiling data (ncu / nsight compute), which has 10-100x overhead
- Bottlenecks can be multi-dimensional (40% memory, 35% compute, 25% latency)
- Bottlenecks shift after optimization (fix memory → now compute-bound)
- Some bottlenecks are invisible at the PTX level (bank conflicts depend on runtime addresses)

**Open sub-question**: Can you diagnose bottlenecks from PTX structure alone (static analysis), or do you need hardware profiling (dynamic analysis)? Is there a middle ground?

### 6.3 Action Generation

**Problem**: Given understanding of the code and its bottleneck, propose a modification that addresses the bottleneck.

**Current state**: 20 validated transforms (B1). Greedy search over combinations gives -28.1% mean. No model-generated actions.

**What makes this hard**:
- L2 actions (current 20 transforms) have a known ceiling (-28.1% mean, -54.3% best)
- Big actions (TMA, shared memory, software pipelining) are hard to validate
- The space of possible modifications is effectively unbounded
- Each new action type needs correctness guarantees (semantic preservation)
- ptxas is a black box — "good" PTX modifications can produce bad SASS

**The key tension**: Constrained actions are safe but limited. Unconstrained actions are expressive but dangerous. The boundary between them is where the engineering meets the research.

**Open sub-question**: Can the action space be structured so that correctness is guaranteed by construction (like the current 4 transforms), while still being expressive enough to capture the "big actions" Aaditya described?

### 6.4 Validation

**Problem**: Given a proposed modification, verify that it (a) produces valid PTX, (b) compiles, (c) gives correct results, (d) is faster.

**Current state**: SM clock() cycle counter (1-cycle precision), ptxas compilation wrapper, correctness checking against reference output. This works.

**What makes this hard**:
- Steps (a)-(c) are cheap (~10ms total) but step (d) is expensive (~50ms)
- For "big actions" that change algorithm structure, correctness checking is harder (different intermediate values, same final result)
- Some bugs are non-deterministic (race conditions from incorrect synchronization)
- False positives: a modification might be faster on one input size but slower on another

**Open sub-question**: Can validation be tiered — cheap static checks killing most invalid proposals, with expensive hardware measurement reserved for the most promising candidates?

### 6.5 Learning

**Problem**: How does the agent improve over time from its optimization experience?

**Current state**: RF trained on 76 (features, best_transform) pairs. No online learning. No fine-tuning.

**What makes this hard**:
- Limited training data (76 kernels is statistically insufficient)
- The optimization landscape changes as the action space expands
- Transfer between kernel types is unclear (does learning to optimize matmul help with attention?)
- The model needs to scale: 4 transforms today, 40 tomorrow, 400 eventually

**Open sub-question**: Is this a fine-tuning problem (pretrained LLM + RLVR), an evolutionary problem (AlphaEvolve-style), a classical RL problem (policy gradient + value function), or something else? Can we distinguish between these empirically before committing?

### 6.6 Generalization

**Problem**: Can the agent optimize kernels it has never seen before? Can it transfer across hardware?

**Current state**: Untested. All evaluation is LOOCV within the 76-kernel dataset.

**What makes this hard**:
- Each kernel type has different bottleneck patterns
- Each hardware architecture has different characteristics
- The agent must generalize across BOTH kernel types AND hardware — a 2D generalization challenge
- Compilers already do a decent job on "standard" patterns — the value is in non-obvious optimizations on unusual kernels

**Open sub-question**: What is the minimum set of diverse kernels needed to test generalization? What constitutes "diverse enough"?

---

## 7. What We Know From Data

Hard facts from Chronos v1 and v2 experiments:

### 7.1 Instruction reordering impact is non-monotonic with kernel size

- gemm_tile 4x4x4 (131 instr): 0.9% headroom — too small for ordering to matter
- gemm_tile 8x8x8 (771 instr): **21.7% speedup** from critical_path ordering
- gemm_tile 16x16x16 (5123 instr): **30.4% regression** from critical_path — hurts at scale
- Mechanism: reordering extends register live ranges → more spills when hitting register ceiling
- At 8x8x8: more registers used (128 vs 80), zero spills → better ILP
- At 16x16x16: all variants hit 255-reg cap, critical_path causes 3.7x more spills (4748 vs 1268 bytes)
- Implication: the agent must learn WHEN to reorder and WHICH strategy, conditioned on kernel size and register pressure

### 7.2 PTX transforms produce real speedups

**Greedy v1 (4 transforms, 9 actions):**
- Up to **50.4%** on individual kernels (gemm 4,6,8 via greedy: maxnreg_128 → vectorize)
- **-20.1% mean** via greedy sequential composition (64 kernels, max 5 steps, avg 2 steps)
- -13.5% mean via landscape oracle (best of 12 fixed configs)
- Greedy: vectorize in 94% of solutions, maxnreg in 71%, reorder in 0%
- 61/64 kernels (95%) improved >1% via greedy composition

**Greedy v2 (8 transform classes, 20 actions) — B1 results:**
- Up to **-54.3%** on individual kernels (gemm 4,6,8: maxnreg_128 → reorder_cp → vec_ld → vec_st)
- **-28.1% mean**, -27.1% median via greedy sequential composition (64 kernels, max 6 steps)
- v2 - v1 delta: **-8.1 percentage points** (from -20.1% to -28.1%)
- vec_st in 88% of solutions, vec_ld in 84%, maxnreg_128 in 31%
- The 8.1pp gain comes primarily from vectorize_stores (-10.2% standalone)

**Common findings:**
- Composition matters: greedy 2-step beats landscape oracle by 49%
- Landscape best config distribution: vec+reorder_il (33%), vectorize (27%), maxnreg_64 (16%)

### 7.3 The policy IS learnable from scalar features

- n_loads ≤ 34: 71% of these kernels prefer vectorize (partially confirmed)
- Best feature for predicting best speedup: fma_ratio (r=0.709), n_loads (r=0.667), n_vregs (r=0.610)
- Best decision stump: `load_ratio ≤ 0.218 → vec+reorder_il, else → maxnreg_128+reorder_cp` captures **86.5% of oracle** savings
- Best single fixed config (maxnreg_128+reorder_cp): captures 61.9% of oracle savings
- Prior claim "RF captures 77% of oracle" not reproduced from this landscape data. Best fixed config = 61.9%, best stump = 86.5%.
- For reorder_cp direction: best predictor is `m` (r=+0.325). Large `m` → reorder helps. But no single feature achieves >60% accuracy.

### 7.4 GNN/LLM collapse is a data problem, not a model problem

- 76 kernels with 50% majority class → guaranteed mode collapse
- GNN DID learn structural discrimination (gemm → vectorize, triton → stop) with diverse data
- 270M LLM as zero-shot predictor with 5 examples is not a fair test of LLM capability

### 7.5 Transforms on Triton PTX — VALIDATED

Tested on 10 diverse Triton kernel types (reduction, scan, attention, activations, cross_entropy, batch_norm, embedding):
- **cache_hints**: Applied to 10/10 kernels. Zero register/spill change. Consistent with gemm_tile.
- **register_budget**: Applied to 10/10 kernels but 3/10 fail to compile (Triton PTX has constraints gemm_tile doesn't). Budget(32) reduces regs by -1.57 avg, budget(128) increases by +2.71 avg.
- **vectorize_loads**: Applied to 0/10 kernels — Triton already vectorizes all loads. CONFIRMED.
- The prior v1/v2 finding "maxnreg_255 gives +5.2% on Triton" needs re-measurement with cycle counter but register changes are directionally consistent.

### 7.6 Non-monotone interactions exist at multiple levels

- **Transform composition**: cache_cg alone: +28% slowdown; cache_cg after vectorize: -0.4% improvement; vec+reorder_cp: ILLEGAL_ADDRESS crash
- **Scale-dependent**: reorder(critical_path): -21.7% at 8x8x8, +30.4% at 16x16x16. Same transform, same strategy, opposite effect at different scales.
- **Register budget**: max_regs=32: +34.5% penalty; max_regs=64: -5.0% improvement. Non-monotone in a single parameter.
- Implication: single-step prediction is insufficient; multi-step reasoning AND scale-awareness are required

### 7.7 Hardware measurement is solved

- SM clock() cycle counter: 1-cycle std dev, 38-sigma significance
- Cost: ~50ms per evaluation (compile + measure)
- This is reliable enough for learning — the noise is in the model, not the measurement

### 7.8 Evaluation cost is feasible

- 768 evaluations (64 kernels × 12 configs) in Phase 6b: minutes
- 64 kernel greedy search: 15 minutes
- 100K evaluations at 50ms each: ~83 minutes
- The bottleneck is kernel diversity, not evaluation speed

---

## 8. What We Don't Know

Questions that MUST be answered empirically before committing to a solution direction:

### 8.1 Model capacity threshold

Does a 270M model have enough capacity to learn PTX → optimization mapping when properly fine-tuned (not just few-shot prompted)? Phase 8c tested zero-shot. We have NO data on fine-tuned performance. This is the single most important unknown.

### 8.2 Data requirements for diverse kernels

How many diverse kernels are needed before learning methods beat the "always maxnreg_128" baseline? Is it 200? 500? 5000? At what diversity threshold does the GNN stop collapsing?

### 8.3 ptxas predictability — PARTIALLY ANSWERED

Landscape data (64 gemm_tile kernels) shows:
- **Cache hints**: Stable and predictable. cg/cv consistently penalize across all kernel sizes. ca/cs are consistently neutral. Effect does NOT depend on kernel shape.
- **Register budget**: Semi-predictable. maxnreg_64 consistently helps for medium kernels (16-48 baseline registers). maxnreg_32 consistently hurts. Occupancy tradeoff is learnable.
- **Reorder**: CHAOTIC. reorder_cp ranges from -18.4% (gemm 8,4,6) to +51.2% (gemm 2,8,8) for similar-sized kernels. The effect depends on kernel SHAPE (m,n,k ratios), not just instruction count. gemm(m,n,k) and gemm(k,n,m) can have opposite reorder effects.
- **Vectorize**: Semi-predictable. Helps 49/64 kernels (77%), but magnitude varies from -30% to +20%.

Implication: learning from PTX-level features IS viable for cache hints and register budget (stable effects). For reorder, the agent needs richer state representation that captures kernel geometry, not just scalar features.

### 8.4 Transfer between kernel types

Does learning to optimize matmul kernels help optimize attention kernels? Or does each kernel type require its own training data? If no transfer, the data requirement multiplies by the number of kernel types.

### 8.5 The "big action" validation cost

How often do big actions (shared memory promotion, software pipelining) produce subtly incorrect code? If 99% of big action proposals are invalid, the effective evaluation cost is 100x the per-evaluation cost. If 50% are valid, it's only 2x.

### 8.6 Profile data value — PARTIALLY ANSWERED

ncu profiling on 68 gemm_tile kernels shows ALL are memory-bound (SM/DRAM ratio 0.04-0.17). Bottleneck classification does NOT differentiate within this dataset — it's a single-thread artifact. The useful ncu data is raw metrics: SASS instruction count (29-5700), FMA count (8-4098), cycle count (24-548), registers (18-255). These correlate with transform effectiveness. For multi-warp production kernels, bottleneck categories may differentiate. Outstanding question: do ncu metrics add predictive power beyond static PTX features?

### 8.7 Evolutionary vs RL vs BC

For this specific problem (PTX optimization with hardware feedback), which learning paradigm converges fastest? No data exists. AlphaEvolve uses evolutionary (works but requires large LLMs). RLVR uses policy gradient (works for math/code verification). BC was tested and failed (but on insufficient data).

### 8.8 The warm-start question — NOW ANSWERABLE

Greedy search generates clean training data:
- **v1**: 64 kernels × (baseline + greedy sequence), 9 actions. Mean -20.1%, best -50.4%. Data: `exp-assembly/data/greedy_search_results.json`
- **v2**: 64 kernels × (baseline + greedy sequence), 20 actions. Mean -28.1%, best -54.3%. Data: `exp-assembly/data/greedy_search_v2_results.json`

This is the BC training signal. If BC clones greedy and then RL/evolutionary pushes beyond it: does the model discover sequences greedy misses? The v2 data with 20 actions provides richer trajectories (avg 3 steps vs 2 steps in v1).

---

## 9. Open Questions

These are higher-level questions that frame the research direction:

### Q1: Is this more like AlphaGo or more like AlphaEvolve?

AlphaGo: single environment (Go), fixed rules, learned from self-play, model-based (value network + policy network + MCTS).
AlphaEvolve: diverse programs, no fixed rules, LLM proposes mutations, evolutionary selection, model-free (no world model).

Kernel optimization is somewhere between:
- Like AlphaGo: there IS a fixed environment (the GPU), and hardware measurement IS the ground truth
- Like AlphaEvolve: each kernel is a different "game" with different structure, and actions are code modifications

Which framing leads to a more tractable system?

### Q2: Where should the model boundary be?

Options:
- **Model sees PTX text**: maximum information, but PTX is verbose and noisy
- **Model sees instruction DAG**: structural information, but loses register names and address patterns
- **Model sees scalar features**: minimal information, but already captures 77% of oracle (RF)
- **Model sees profile data**: bottleneck information, but expensive to collect
- **Model sees PTX + profile**: maximum information, expensive

What's the minimum state representation that enables the agent to make good decisions?

### Q3: What's the right evaluation protocol?

Current: LOOCV on 76 kernels. This tests interpolation, not generalization.

What we need: held-out kernel FAMILIES. Train on matmul + reduction + elementwise. Test on attention + convolution. This tests whether the agent learned general optimization principles, not kernel-specific patterns.

But we don't have enough kernel families yet to do this.

### Q4: How do you grow the action space safely?

Start with 20 validated transforms (B1, up from 4). How do you:
- Add new transforms without retraining from scratch?
- Test that new transforms compose safely with existing ones?
- Gradually give the model more expressive actions (from fixed transforms to parameterized transforms to diffs)?

### Q5: What's the relationship between the Chronos transform library and compiler passes?

Are Chronos transforms doing the same thing as LLVM optimization passes? If so, we're reimplementing the compiler. If not, what's different?

Understanding where the current transforms sit relative to existing compiler infrastructure tells us whether we're adding value or duplicating effort.

### Q6: What can we learn from AlphaEvolve's failures?

AlphaEvolve worked on GEMM configuration heuristics. It did NOT work on arbitrary code rewriting. The 23% speedup came from discovering better tile/block size selection logic, not from rewriting assembly.

Is PTX-level code rewriting fundamentally harder than configuration optimization? Why?

### Q7: What does Aaditya's company (Lossfunk) actually need?

The theoretical problem (agent that replaces kernel engineers) is unbounded. The practical problem (optimize specific kernels for specific hardware for a specific product) is bounded. Understanding the business constraint focuses the research.

---

## 10. Research TODO

### Phase A: Ground Truth Collection

- [x] **A1**: Extract PTX from 10 diverse real-world sources — 15 kernel types, 33 variants (21 new + 12 existing). PyTorch inductor confirmed viable (5/6 ops produce PTX)
- [x] **A2**: Landscape sweep COMPLETE. 64 kernels × 12 configs (existing data validated). Extended to 10x10x10, 12x12x12, 14x14x14. Key finding: register headroom determines which transforms work. At 14x14x14 (255 regs, already spilling), NO transform helps.
- [x] **A3**: Greedy search COMPLETE. 64 kernels, sequential composition. Mean improvement: -20.1% (vs -13.5% landscape oracle). Best: -50.4%. Vectorize in 94% of solutions. Dominant pattern: maxnreg → vectorize (2 steps). NO reorder chosen by greedy.
- [x] **A4**: ncu profiling COMPLETE. 68 kernels profiled (64 base + 4 extended). ALL kernels are memory-bound (SM/DRAM ratio 0.04-0.17). Only 14x14x14 and 16x16x16 show local memory spills. ncu 2024.3.2 installed at /opt/nvidia/nsight-compute/2024.3.2/ncu.
- [x] **A5**: ptxas sensitivity COMPLETE — ~~ptxas ignores instruction ordering~~ **CORRECTED**: reorder impact is non-monotonic (-21.7% at 8x8x8, +30.4% at 16x16x16). Cache hints pass through 1:1. Vectorize loads causes genuine structural change. See Section 12 for full revised findings.
- [x] **A6**: Hardware cycle measurement for all transforms + reorder scaling (4x4x4, 8x8x8, 16x16x16). Total cycle range: 597-26428. Register spills identified as mechanism for 16x16x16 regression.

### Phase B: Empirical Unknowns

- [x] **B1**: Expand action space from 9 to 20 validated transforms. Greedy v2: **-28.1% mean** (up from -20.1% v1, +8.1pp). vectorize_stores is as impactful as vectorize_loads. See Section 13.
- [ ] **B1b**: Fine-tune CodeGemma 270M / 2B on existing 76-kernel data (proper training, not few-shot). Compare to RF baseline. Answer: does fine-tuning help at all?
- [ ] **B2**: Measure GNN performance as a function of dataset size: train on 20, 40, 60, 76 kernels, plot learning curve. Answer: is the GNN data-limited or model-limited?
- [ ] **B3**: Test transfer: train on gemm-only, evaluate on Triton kernels. Train on Triton-only, evaluate on gemm. Answer: does cross-kernel-type transfer exist?
- [ ] **B4**: Implement one "big action" (loop unrolling or shared memory promotion). Measure validation failure rate on 20 kernels. Answer: how often do big actions produce invalid code?
- [ ] **B5**: Compare evolutionary (OpenEvolve-style) vs greedy on the same kernel set. Answer: does evolutionary search find better transform sequences?

### Phase C: Problem Characterization

- [~] **C1**: PARTIALLY DONE. ncu profiled 68 kernels — ALL are memory-bound (single-thread artifact). Bottleneck classification does NOT differentiate within this dataset. Raw ncu metrics (cycles, instruction counts) are useful as continuous features. Full answer requires profiling multi-warp production kernels.
- [ ] **C2**: Analyze ptxas SASS output for all 76 kernels + their best transforms. Answer: do transforms change SASS in predictable ways, or is ptxas chaotic?
- [x] **C3**: ANSWERED by B1. Ceiling with 20 transforms: **-28.1% mean, -54.3% best** (vs 4 transforms: -20.1% mean, -50.4% best). The +8.1pp gain came mostly from vectorize_stores. Remaining new transforms (prefetch, store cache hints, split vector loads, new reorder strategies) contributed marginally in isolation but appear in ~5/64 greedy solutions in composition.
- [ ] **C4**: Study AlphaEvolve, CodeEvolve, PEAK papers in detail. Extract: action representations, evaluation protocols, data requirements, failure modes. Answer: what can we reuse?
- [ ] **C5**: Study CuAsmRL (CGO 2025) and CompilerDream papers. Answer: what did SASS-level and LLVM-level RL approaches learn that we can apply?

### Phase D: Infrastructure

- [ ] **D1**: Build PTX extraction pipeline for PyTorch inductor kernels (compile a model, intercept PTX)
- [ ] **D2**: Build PTX extraction pipeline for CUTLASS templates (compile with different parameters)
- [ ] **D3**: Standardize the data format: (kernel_id, ptx_source, kernel_type, instruction_count, ncu_profile, transform_applied, speedup)
- [ ] **D4**: Build a diff-based action representation (like AlphaEvolve's structured diffs for PTX)
- [ ] **D5**: Set up fine-tuning pipeline for CodeGemma / small LLM on optimization data

---

## 11. Phase A Research Results (2026-02-16)

### A1: Diverse Kernel Extraction — COMPLETE

**Script**: `exp-assembly/scripts/generate_diverse_kernels.py`
**Data**: `exp-assembly/data/diverse_kernel_catalog.json`

Generated and compiled 21 kernel variants across 10 new kernel types (beyond existing 5), all targeting sm_89 (L4 GPU):

| Kernel Type | Variants | Pattern | Instr Range | Shared Mem | Barriers |
|---|---|---|---|---|---|
| reduction_sum | 2 | reduction | 88-92 | Yes | Yes |
| reduction_max | 2 | reduction | 88-92 | Yes | Yes |
| prefix_scan | 2 | scan | 198-234 | Yes | Yes |
| attention | 3 | attention | 1584-5564 | Yes | Yes |
| relu | 2 | elementwise | 40-72 | No | No |
| gelu | 2 | elementwise | 100-192 | No | No |
| dropout | 2 | elementwise_random | 344-648 | No | No |
| cross_entropy | 2 | reduction_loss | 240-252 | Yes | Yes |
| batch_norm | 2 | normalization | 146-146 | No | No |
| embedding_lookup | 2 | gather | 30-42 | No | No |

**Result**: 21/21 compiled, 0 failures. Combined with existing 5 types (vector_add, softmax, layernorm, matmul, fused_add_mul) = **15 kernel types, 33 total variants**. Instruction count spans 200x range (30 to 5564). 8 distinct computational patterns.

### A1b: PyTorch Inductor PTX Extraction — CONFIRMED VIABLE

**Script**: `exp-assembly/scripts/test_inductor_ptx.py`

| Operation | PTX? | #PTX files | PTX bytes | Kernel Type |
|---|---|---|---|---|
| matmul | NO | 0 | 0 | cublas (no Triton PTX) |
| elementwise | YES | 2 | 7,501 | triton |
| reduction | YES | 1 | 5,439 | triton |
| softmax | YES | 1 | 9,877 | triton |
| layernorm | YES | 3 | 33,600 | triton |
| mlp | YES | 2 | 7,594 | triton + cublas mixed |

**Key findings**:
- **5/6 operations produce extractable PTX** (matmul routes to cuBLAS)
- All PTX is Triton-generated via LLVM NVPTX backend
- Full pipeline cached: `.py` → `.ttir` → `.ttgir` → `.llir` → `.ptx` → `.cubin`
- Inductor fuses operations (relu+add → single kernel, softmax → single kernel)
- Config: `inductor_config.triton.store_cubin = True` + `trace.enabled = True`
- Descriptive names: `triton_poi_fused_add_relu_0.ptx`, `triton_per_fused__softmax_0.ptx`

**Conclusion**: PyTorch inductor is a viable, scalable PTX source. Any PyTorch model can generate diverse kernels.

### A3: Greedy Sequential Search — UPPER BOUND DATA

**Script**: `exp-assembly/scripts/greedy_search.py`
**Data**: `exp-assembly/data/greedy_search_results.json`

Greedy search on 64 gemm_tile kernels. For each kernel: try each transform, keep the best, repeat until no improvement. Max 5 steps.

| Metric | Value |
|---|---|
| Mean improvement | **-20.1%** |
| Median improvement | -15.1% |
| Best | **-50.4%** (gemm_tile 4,6,8) |
| Kernels improved (>1%) | 61/64 (95%) |
| Mean steps | 2.0 |
| Most common step count | 2 steps (59% of kernels) |

**Transform frequency in greedy solutions:**

| Transform | Frequency | Role |
|---|---|---|
| vectorize | 60/64 (94%) | Nearly universal first or second step |
| maxnreg_128 | 23/64 (36%) | Dominant register budget for medium kernels |
| cache_cs | 14/64 (22%) | Mild safe addition |
| maxnreg_64 | 12/64 (19%) | For smaller kernels |
| maxnreg_255 | 10/64 (16%) | For larger kernels |
| cache_cg | 9/64 (14%) | Sometimes helps in composition |
| maxnreg_32 | 3/64 (5%) | Rare, for tiny kernels |
| **reorder** | **0/64 (0%)** | **Never selected by greedy** |

**Dominant composition patterns:**

| Pattern | Count | Example |
|---|---|---|
| maxnreg_128 → vectorize | 11 | gemm_tile(4,6,8): -50.4% |
| vectorize (alone) | 6 | gemm_tile(4,8,2): -5.2% |
| vectorize → maxnreg_128 | 6 | gemm_tile(6,8,6): -24.7% |
| vectorize → cache_cg | 4 | gemm_tile(6,8,2): -14.3% |
| maxnreg_64 → vectorize → cache_cs | 4 | gemm_tile(2,8,4): -31.4% |

**Greedy vs landscape comparison:**
- Landscape oracle (best of 12 fixed configs): **-13.5%** mean
- Greedy sequential composition: **-20.1%** mean (49% better than oracle)
- Greedy composition finds sequences the landscape sweep cannot explore

**Why reorder never appears in greedy:** The greedy algorithm tests reorder at each step. Even though reorder_cp gives -21.7% standalone at 8x8x8, when vectorize or register budget is applied first, they change the register pressure enough that reorder no longer helps (or hurts). Composition ordering matters: vectorize first reduces instructions, then register budget fills headroom — reorder on top of that cannot add value.

### A4: ncu Profiling — BOTTLENECK CLASSIFICATION

**Script**: `exp-assembly/scripts/collect_ncu_profiles.py`
**Data**: `exp-assembly/data/ncu_profiles.json`

ncu 2024.3.2 profiling on 68 kernels (64 base + 4 extended: 10x10x10, 12x12x12, 14x14x14, 16x16x16).

**Key finding: ALL 68 kernels are memory-bound.**

| Metric | Range across 68 kernels |
|---|---|
| SM throughput | 0.01% - 0.15% of peak |
| DRAM throughput | 0.28% - 1.13% of peak |
| SM/DRAM ratio | 0.04 - 0.17 (memory-bound threshold: <0.67) |
| Registers/thread | 18 - 255 |
| SASS instructions | 29 - 5700 |
| FMA instructions | 8 - 4098 |
| Cycles active | 23 - 548 |

**Spill indicator (local memory):**
- 66/68 kernels: **zero local memory access** (no spills)
- gemm_tile(14,14,14): local_loads=76, local_stores=76 (mild spills)
- gemm_tile(16,16,16): local_loads=317, local_stores=317 (heavy spills)

**Bottleneck uniformity:** All single-thread gemm_tile kernels are memory-bound because they run on a single warp with 1 thread — SM utilization is inherently low. This means:
1. ncu bottleneck classification does NOT differentiate between kernels in this dataset
2. The value of ncu data is in the absolute metrics (cycles, instruction counts), not bottleneck type
3. For real multi-warp kernels (Triton, production GEMM), bottleneck classification would be more varied
4. The agent's feature set should include ncu metrics as raw values, not as bottleneck categories

### A5: ptxas Sensitivity Analysis — CRITICAL FINDINGS

**Script**: `exp-assembly/scripts/test_ptxas_sensitivity.py` (775 lines, 3-level diffing)

Tested on gemm_tile(4,4,4) baseline (132 PTX instructions → 121 SASS instructions, 40 registers):

| Transform | PTX chg | Norm SASS % | Opcode % | Regs | Classification |
|---|---|---|---|---|---|
| register_budget(max_regs=32) | 3 | 81.8% | 38.0% | 40→32 | CHAOTIC |
| register_budget(max_regs=64) | 3 | 46.3% | 22.3% | 40→48 | DISRUPTIVE |
| register_budget(max_regs=128) | 3 | 46.3% | 22.3% | 40→48 | DISRUPTIVE |
| register_budget(max_regs=255) | 3 | 46.3% | 22.3% | 40→48 | DISRUPTIVE |
| cache_hints(policy=cs) | 32 | 26.4% | 26.4% | 40→40 | MODERATE |
| cache_hints(policy=cg) | 32 | 26.4% | 26.4% | 40→40 | MODERATE |
| cache_hints(policy=ca) | 32 | 26.4% | 26.4% | 40→40 | MODERATE |
| cache_hints(policy=cv) | 32 | 26.4% | 26.4% | 40→40 | MODERATE |
| reorder(critical_path) | 163 | 43.0% | 18.2% | 40→40 | DISRUPTIVE |
| reorder(interleave) | 165 | 43.0% | 18.2% | 40→40 | DISRUPTIVE |
| vectorize_loads(all) | 32 | 100.8% | 76.9% | 40→40 | CHAOTIC |

Three levels of diffing used:
- **RAW**: exact SASS string match (inflated by register renaming)
- **NORMALIZED**: register names replaced with placeholders (structural diff)
- **OPCODE**: only instruction types and ordering (ignoring operands)

#### Finding 1: ptxas sensitivity to instruction ordering is scale-dependent

At 131 PTX instructions (4x4x4), critical_path and interleave produce identical SASS to each other (both differ from baseline by 8.2% opcode). At 771 instructions (8x8x8), all three variants produce genuinely different SASS with different instruction counts, yielding a 21.7% cycle improvement for critical_path. At 5123 instructions (16x16x16), critical_path causes 3.7x more register spills and a 30.4% cycle regression. See Section 12 for full data.

#### Finding 2: Cache hints are STABLE and pass through 1:1

32 PTX loads hinted → 32 SASS instructions changed (LDG.E → LDG.E.CS/CG/CA/CV). Amplification = 1.0x. No register renaming, no reordering. The agent can learn cache hint effects directly.

#### Finding 3: Register budget causes register renaming cascades

Changing max_regs from 40→32 or 40→48 causes ptxas to reallocate all registers. Raw SASS diff is 96-100%, but opcode diff is only 22-38%. The instruction MIX barely changes; only register assignments change. Implication: features must be register-name-independent.

#### Finding 4: Vectorize loads causes genuine structural changes

Fusing 32 scalar loads into 8 vector loads (LDG.E.128) changes 76.9% of opcodes. ptxas generates fundamentally different instruction sequences. This is the kind of transform the agent should learn — semantic changes that alter the instruction mix.

#### Transform Categories (Validated with Cycle Measurements)

| Category | Effect on SASS | Cycle Impact | Agent Relevance |
|---|---|---|---|
| **Instruction ordering** (reorder) | Scale-dependent: 8%→33%→67% opcode diff | -21.7% (8x8x8), +30.4% (16x16x16) | HIGH but non-monotonic — must learn when to apply |
| **Cache hints** | 1:1 pass-through, stable | cg/cv: +32.3%, ca/cs: 0% | Moderate — can hurt badly if wrong |
| **Register budget** | Register renaming cascade | -5.0% to +34.5% | Moderate — occupancy vs spill tradeoff |
| **Vectorize loads** | Genuine structural change | -14.2% | HIGH — changes instruction mix |
| **New transforms needed** | Unknown | Unknown | Research frontier |

### Phase A Status Summary

| Task | Status | Finding |
|---|---|---|
| A1: Extract diverse PTX | DONE | 15 types, 33 variants, 30-5564 instr range |
| A1b: Inductor extraction | DONE | Works for 5/6 ops, full pipeline cached |
| A2: Landscape sweep | DONE | 64 kernels x 12 configs + extended to 10/12/14. Register headroom determines transform effectiveness. |
| A3: Greedy search | DONE | Mean -20.1%, best -50.4%. Vectorize in 94% of solutions. 2-step composition is dominant. No reorder in any greedy solution. |
| A4: ncu profiling | DONE | 68 kernels profiled. ALL memory-bound (SM/DRAM ratio 0.04-0.17). Only 14x14x14+ show spills. |
| A5: ptxas sensitivity | DONE | Reorder is scale-dependent (-21.7% at 8x8x8, +30.4% at 16x16x16), cache hints stable, vectorize structural |
| A6: Cycle measurements | DONE | All transforms produce measurable cycle diffs. Total range: 597-26428 cycles. Spills identified as mechanism. |

### Implications for Chronos v3

Based on validated cycle measurements across kernel sizes:

1. **Instruction reordering is a primary action for medium kernels (500-1000 instr).** Critical_path ordering gives 21.7% speedup at 771 instructions. But at 5123 instructions, the same strategy causes 30.4% regression due to register spills. The agent must learn scale-dependent policies.

2. **Cache hints are a safe starting action** — predictable, stable, 1:1 pass-through. But cg/cv carry +32.3% penalty — the agent must learn which policies help and which hurt.

3. **Vectorize loads is the highest-impact single transform** (-14.2% on 4x4x4). It changes the instruction mix in ways ptxas must respect.

4. **Feature representations must be register-independent**. Raw SASS diffs are misleading because register renaming dominates. Use opcode-only or structural features.

5. **The action space spans both instruction scheduling and semantic rewrites.** Reordering alone gives -21.7% at medium scale. Vectorization gives -14.2%. Semantic rewrites (shared memory promotion, TMA insertion, loop unrolling) are the research frontier.

6. **Hardware validation is the ground truth.** Every SASS-level change produces a measurable cycle difference. The SM cycle counter (1-cycle std dev on small kernels, <0.25% relative noise on large kernels) is the verifier.

## 12. Phase A Revision: Reorder IS Performance-Relevant (2026-02-16, session 2)

### CORRECTION: The "reorder is a no-op" finding from Section 11 is WRONG for larger kernels

The original A5 experiment tested only gemm_tile(4,4,4) with 131 PTX instructions. At that scale, critical_path and interleave reorders produce identical SASS (compared to each other). This led to the incorrect conclusion that ptxas ignores instruction ordering entirely.

**Scaling experiments disprove this.** At 771+ PTX instructions, ptxas produces genuinely different SASS for different instruction orderings, with measurable cycle count differences.

### A5 Corrected: Reorder SASS Analysis Across Kernel Sizes

**Scripts**: `exp-assembly/scripts/test_reorder_scaling.py` (fixed SASS parsing + cubin hash comparison)

| Kernel | PTX instr | SASS (base) | SASS (cp) | SASS (il) | cp vs base (opcode%) | il vs base (opcode%) | cp vs il (opcode%) |
|--------|-----------|-------------|-----------|-----------|---------------------|---------------------|-------------------|
| 4x4x4 | 131 | 122 | 122 | 122 | 8.2% | 8.2% | **0.0% (identical)** |
| 8x8x8 | 771 | 795 | 742 | 737 | **33.0%** | **41.3%** | 21.7% |
| 16x16x16 | 5123 | 5701 | 7394 | 7029 | **66.7%** | **63.5%** | 56.6% |

Key observations:
- At 4x4x4: critical_path == interleave (0% diff between them), but both differ from baseline (8.2% opcode)
- At 8x8x8: ALL THREE variants produce genuinely different SASS with different instruction counts (795 vs 742 vs 737)
- At 16x16x16: differences are massive. critical_path produces **30% more SASS instructions** than baseline (7394 vs 5701)
- The opcode diff scales: 8.2% → 33% → 67% as kernel size grows from 131 → 771 → 5123 PTX instructions
- CUBIN hashes confirm: all variants at 8x8x8 and 16x16x16 are byte-different binaries
- At 16x16x16, even the two reorder strategies diverge substantially (56.6% opcode diff between them)

**WARNING**: At 16x16x16, critical_path ordering produces 7394 SASS instructions vs 5701 baseline — a 30% increase. This suggests that the DAG-based critical_path heuristic, while beneficial at 8x8x8 (-21.6% cycles), may be counterproductive at larger scales where it confuses ptxas's internal scheduler. The relationship between PTX ordering and SASS quality is non-monotonic and kernel-size-dependent.

### Hardware Cycle Measurements: Reorder Impact

**Method**: SM clock() instrumentation, 50 warmup + 200 measurement runs, subprocess-isolated

| Kernel | Variant | Median Cycles | Std | Delta vs Baseline |
|--------|---------|--------------|-----|-------------------|
| 4x4x4 | baseline | 697 | 1.1 | - |
| 4x4x4 | reorder(critical_path) | 691 | 1.3 | **-0.9%** |
| 4x4x4 | reorder(interleave) | 691 | 1.2 | **-0.9%** |
| 8x8x8 | baseline | 2452 | 1.3 | - |
| 8x8x8 | reorder(critical_path) | 1921 | 1.5 | **-21.7%** |
| 8x8x8 | reorder(interleave) | 2534 | 2.0 | **+3.3%** |
| 16x16x16 | baseline | 20272 | 8.6 | - |
| 16x16x16 | reorder(critical_path) | 26428 | 12.7 | **+30.4%** |
| 16x16x16 | reorder(interleave) | 25017 | 47.3 | **+23.4%** |

**Script**: `exp-assembly/scripts/test_reorder_cycles_scaling.py`
**Data**: `exp-assembly/data/reorder_cycle_scaling.json`

At 8x8x8: critical_path gives a **21.7% speedup** (531 fewer cycles). Interleave gives a 3.3% regression. Spread = 613 cycles.

At 16x16x16: **BOTH reorderings hurt performance.** Critical_path = +30.4% (6,156 MORE cycles), interleave = +23.4% (4,745 MORE cycles). The baseline (original PTX ordering from the template generator) is the best variant. The 30% more SASS instructions from critical_path map directly to 30% more cycles — the GPU does NOT compensate.

Std dev scales with kernel size: 1.3-2.0 at 8x8x8, 8.6-47.3 at 16x16x16. The interleave variant at 16x16x16 has notably high variance (47.3 std dev), suggesting it produces less deterministic execution.

**Register/spill analysis — the mechanism behind the regression:**

| Kernel | Variant | Registers | Spill Stores (bytes) | Spill Loads (bytes) |
|--------|---------|-----------|---------------------|-------------------|
| 8x8x8 | baseline | 80 | 0 | 0 |
| 8x8x8 | critical_path | 128 | 0 | 0 |
| 8x8x8 | interleave | 96 | 0 | 0 |
| 16x16x16 | baseline | 255 | 1268 | 1268 |
| 16x16x16 | critical_path | 255 | **4748** | **4748** |
| 16x16x16 | interleave | 255 | **4060** | **4060** |

At 8x8x8: critical_path makes ptxas use 128 registers (vs 80 baseline) — more registers enable better scheduling, no spills, hence the 21.7% speedup. At 16x16x16: all variants hit the 255-register ceiling. critical_path causes **3.7x more register spills** (4748 vs 1268 bytes). The DAG-based ordering creates longer live ranges that force ptxas to spill more values to local memory. Each spill store/load pair costs ~20-100 cycles depending on cache state. This is the direct mechanism: more spills → more memory traffic → 30.4% slower.

### Hardware Cycle Measurements: All Transforms on 4x4x4

**Script**: `exp-assembly/scripts/test_transform_cycles.py`
**Data**: `exp-assembly/data/transform_cycle_results.json`

| Transform | Median Cycles | Std | Regs | Delta vs Baseline |
|-----------|--------------|-----|------|-------------------|
| baseline | 696 | 1.1 | 40 | - |
| cache_hints(ca) | 696 | 1.2 | 40 | +0.0% |
| cache_hints(cs) | 696 | 1.1 | 40 | +0.0% |
| cache_hints(cg) | 921 | 1.6 | 40 | **+32.3%** |
| cache_hints(cv) | 921 | 1.4 | 40 | **+32.3%** |
| reg_budget(32) | 936 | 1.6 | 32 | **+34.5%** |
| reg_budget(64) | 661 | 1.2 | 48 | **-5.0%** |
| reg_budget(128) | 661 | 1.1 | 48 | **-5.0%** |
| vectorize_loads | 597 | 1.2 | 40 | **-14.2%** |

Total cycle range: 597-936 (48.7% spread). All variants produce correct results.

### Hardware Cycle Measurements: All Transforms on 8x8x8

**Script**: `exp-assembly/scripts/test_all_transforms_8x8x8.py`
**Data**: `exp-assembly/data/transform_cycles_8x8x8.json`

| Transform | Median Cycles | Std | Regs | Spills (S/L) | Delta vs Baseline |
|-----------|--------------|-----|------|-------------|-------------------|
| baseline | 2444 | 1.1 | 80 | 0/0 | - |
| cache_hints(ca) | 2452 | 1.2 | 80 | 0/0 | +0.3% |
| cache_hints(cs) | 2452 | 1.2 | 80 | 0/0 | +0.3% |
| cache_hints(cg) | 2569 | 1.8 | 80 | 0/0 | **+5.1%** |
| cache_hints(cv) | 2569 | 1.8 | 80 | 0/0 | **+5.1%** |
| reg_budget(32) | 6729 | 2.4 | 32 | 684/784 | **+175.3%** |
| reg_budget(64) | 3325 | 2.0 | 64 | 168/168 | **+36.0%** |
| reg_budget(128) | 1997 | 1.9 | 122 | 0/0 | **-18.3%** |
| reg_budget(255) | 1875 | 3.9 | 167 | 0/0 | **-23.3%** |
| vectorize_loads | 2261 | 1.3 | 96 | 0/0 | **-7.5%** |
| reorder_cp | 1921 | 1.5 | 128 | 0/0 | **-21.4%** |
| reorder_il | 2525 | 2.1 | 96 | 0/0 | +3.3% |
| vec+reorder_il | 1758 | 1.2 | 149 | 0/0 | **-28.1%** |
| maxnreg_128+reorder_cp | 1937 | 1.2 | 128 | 0/0 | **-20.8%** |

**Transform effects are NOT consistent across kernel sizes.** Comparing 4x4x4 vs 8x8x8:
- cache_cg penalty: +32.3% → +5.1% (penalty shrank 6x)
- reg_budget(64): **-5.0% → +36.0%** (flipped from improvement to regression! Causes 168B spills at 8x8x8)
- reg_budget(128): -5.0% → **-18.3%** (improvement grew 3.6x)
- reg_budget(255): N/A → **-23.3%** (new best single transform for 8x8x8)
- vectorize: -14.2% → -7.5% (still helps but less)
- reorder_cp: -0.9% → **-21.4%** (improvement grew 24x)
- **vec+reorder_il: -28.1%** (best composite, not tested on 4x4x4)

The mechanism is register pressure. At 8x8x8 (baseline 80 regs), there's room for ptxas to use more registers (up to 167 for reg_budget(255)). More registers → less spilling → better ILP. But reg_budget(64) now CAUSES spills (168 bytes) because 64 regs is too tight for an 80-reg kernel.

### Extended Landscape: 10x10x10, 12x12x12, 14x14x14

**Script**: `exp-assembly/scripts/sweep_extended.py`
**Data**: `exp-assembly/data/landscape_extended.json`

| Kernel | PTX Instr | Base Regs | Base Spills | Best Transform | Best Delta | Worst Transform | Worst Delta |
|--------|-----------|-----------|-------------|---------------|-----------|----------------|------------|
| 10x10x10 | 1403 | 168 | 0 | vectorize | **-15.4%** | maxnreg_32 | +330.9% |
| 12x12x12 | 2307 | 254 | 0 | vectorize | **-22.3%** | maxnreg_32 | +387.1% |
| 14x14x14 | 3531 | 255 | 304 | cache_cs | **-0.0%** | maxnreg_32 | +734.2% |

The **register ceiling** determines which transforms work:

| Baseline Regs | Register Headroom | Best Strategy | Max Improvement |
|--------------|-------------------|--------------|-----------------|
| 40 (4x4x4) | 215 regs free | vectorize alone | -14.2% |
| 80 (8x8x8) | 175 regs free | vec+reorder_il | -28.1% |
| 168 (10x10x10) | 87 regs free | vectorize | -15.4% |
| 254 (12x12x12) | 1 reg free | vectorize | -22.3% |
| 255 (14x14x14) | **0 regs free, already spilling** | nothing | -0.0% |

At 14x14x14, the kernel already uses max registers and spills 304 bytes. No PTX-level transform can help — the kernel needs algorithmic restructuring (tiling, shared memory, TMA) to reduce register pressure. This is the ceiling of the current transform set.

### Key Findings (Validated Across Kernel Sizes and Types)

**1. ALL transform effects are scale-dependent.** No transform has a fixed effect. Every transform's cycle impact changes with kernel size:

| Transform | 4x4x4 (131 instr) | 8x8x8 (771 instr) | Direction |
|-----------|-------------------|-------------------|-----------|
| cache_cg/cv | +32.3% | +5.1% | Penalty shrank 6x |
| reg_budget(32) | +34.5% | +175.3% | Penalty grew 5x (spills) |
| reg_budget(64) | -5.0% | **+36.0%** | **Flipped sign** |
| reg_budget(128) | -5.0% | -18.3% | Improvement grew 3.6x |
| reg_budget(255) | N/A | -23.3% | Best single transform at 8x8x8 |
| vectorize | -14.2% | -7.5% | Improvement shrank 2x |
| reorder_cp | -0.9% | -21.4% | Improvement grew 24x |
| vec+reorder_il | N/A | **-28.1%** | Best composite at 8x8x8 |

**2. The mechanism is always register pressure.** Every transform's effect traces back to how it changes register allocation:
   - reorder extends live ranges → more registers needed → helps when headroom exists, hurts when hitting ceiling
   - reg_budget directly controls register cap → too low = spills, high = better ILP
   - vectorize fuses loads → fewer instructions → fewer registers needed → frees headroom
   - cache hints change memory behavior but NOT register allocation (consistent effect across scales)

**3. Reorder impact depends on kernel SHAPE, not just size.** Landscape data (64 kernels) shows reorder_cp ranges from -51.2% (gemm 2,8,8) to +18.4% (gemm 8,4,6) for kernels of similar instruction count. The best predictor is `m` (r=+0.325): large `m` → reorder helps. But no single feature achieves >60% accuracy. The effect is chaotic for fixed heuristics.

**4. Best decision stump captures 86.5% of oracle.** `load_ratio ≤ 0.218 → vec+reorder_il, else → maxnreg_128+reorder_cp`. This simple rule outperforms any fixed config (best fixed: 61.9% of oracle). The policy IS learnable.

**5. Vectorize does NOT apply to Triton kernels.** Triton already vectorizes loads. 0/10 diverse Triton kernels had applicable vectorize targets. This means the agent's action space is kernel-source-dependent.

**6. Cache hints are the most predictable transform.** ca/cs = consistent neutral. cg/cv = consistent penalty (shrank from +32.3% to +5.1% at 8x8x8 but always negative). Low variance across kernels.

**7. vec+reorder_il is the dominant composite in the landscape.** Best config for 33% of 64 kernels in the 12-config sweep, -28.1% on 8x8x8. But greedy search (which explores sequential compositions) never selects reorder — it finds better sequences.

**8. Greedy composition beats landscape oracle by 49% (v1) and 108% (v2).** Greedy v1 achieves -20.1% mean vs -13.5% landscape oracle. Greedy v2 (20 transforms) achieves **-28.1% mean** — 108% better than landscape oracle. Vectorize_loads appears in 84% and vectorize_stores in 88% of v2 solutions. The dominant v2 pattern includes vec_ld + vec_st + maxnreg in most solutions.

**9 (NEW). vectorize_stores is as impactful as vectorize_loads.** Standalone: vec_st -10.2% vs vec_ld -9.4%. In greedy v2: vec_st in 88% of solutions vs vec_ld in 84%. This is the single largest contributor to the v1→v2 improvement. The store vectorization pattern mirrors load vectorization: merge consecutive `st.global.f32` to same base register into `st.global.v2/v4.f32`.

**9. ALL gemm_tile kernels are memory-bound per ncu.** SM/DRAM throughput ratio ranges 0.04-0.17 across 68 kernels. This is a dataset artifact: single-thread kernels with 1 warp have inherently low SM utilization. ncu bottleneck classification does not differentiate within this dataset. For the agent, raw ncu metrics (instruction counts, cycle counts) are more informative than categorical bottleneck labels.

### Transform Categories (Validated at Multiple Scales)

| Category | 4x4x4 Effect | 8x8x8 Effect | Scale Behavior | Agent Action |
|----------|-------------|-------------|---------------|--------------|
| **reorder_cp** | -0.9% | -21.4% | Grows then reverses at 16x16x16 (+30.4%) | Learn WHEN to apply |
| **reorder_il** | -0.9% | +3.3% | Mild regression | Avoid standalone |
| **vec+reorder_il** | N/A | **-28.1%** | Dominant composite | Default composite |
| **reg_budget(128+)** | -5.0% | **-18.3 to -23.3%** | Grows with kernel register demand | When baseline regs > 60 |
| **reg_budget(32/64)** | -5% to +34.5% | +36% to +175% | Gets worse as kernel grows | Avoid for large kernels |
| **vectorize** | -14.2% | -7.5% | Diminishes with size | Good for small/medium |
| **cache_cg/cv** | +32.3% | +5.1% | Penalty shrinks but persists | Avoid |
| **cache_ca/cs** | 0% | +0.3% | Neutral at all sizes | Safe no-op |

### What This Means for Chronos v3

1. **The agent must condition on kernel size/register pressure.** Fixed configs fail. Even the best decision stump (86.5% of oracle) only works within the {2,4,6,8}^3 range. Real kernels span 30-5000+ instructions.

2. **Composition is more important than transform selection.** Greedy v2 multi-step composition (-28.1% mean) beats best fixed config (-13.5% oracle) by 108%. The agent should learn SEQUENCES, not single actions. The dominant v2 pattern involves vectorize_loads + vectorize_stores + register budget.

3. **Vectorize (loads AND stores) is the universal action.** vec_ld present in 84%, vec_st in 88% of greedy v2 solutions. The agent should try both vectorize directions on every kernel.

4. **Register budget is the primary knob.** maxnreg_128 (36% of solutions), maxnreg_255 (16%), and maxnreg_64 (19%) cover 71% of greedy solutions. The right budget depends on baseline register count.

5. **Reorder is dominated by composition.** Despite standalone reorder_cp giving -21.7% at 8x8x8, greedy NEVER selects reorder. When vectorize + register budget are applied first, they change the register landscape enough that reorder adds no value. The agent should deprioritize reorder in favor of vectorize + register budget.

6. **The greedy ceiling is -54.3%.** Best achieved by maxnreg_128 → reorder_cp → vec_ld → vec_st on gemm_tile(4,6,8) (v2). Up from -50.4% (v1). Beyond this, the agent needs L3+ transforms (shared memory promotion, loop unrolling, TMA) to push further.

7. **ncu profiling is useful for raw metrics, not bottleneck categories.** All single-thread kernels are memory-bound. The agent should use ncu cycle counts and instruction counts as continuous features, not categorical bottleneck labels.

8. **The non-monotonic finding is the key insight.** The relationship between PTX ordering and SASS quality is not monotonic:
   - 131 instr: reorder barely matters (-0.9%)
   - 771 instr: critical_path is optimal (-21.7%)
   - 5123 instr: baseline (no reorder) is optimal, critical_path hurts (+30.4%)
   But greedy search shows this is moot — composition with vectorize + register budget dominates reorder entirely.

---

## 13. Phase B1 Results: Expanded Action Space (2026-02-16)

### B1 Goal

Expand the transform library from 4 classes (9 actions) to 8 classes (20 actions) to test whether more L2 transforms improve greedy search performance. This directly answers C3 (ceiling with more transforms) and provides richer training data for RLVR.

### New Transform Classes

| Class | File | Actions Added | Description |
|-------|------|---------------|-------------|
| VectorizeStoresTransform | `transform/vectorize_stores.py` | vec_st | Merge consecutive `st.global.f32` to same base register into `st.global.v2/v4.f32` |
| PrefetchTransform | `transform/prefetch.py` | prefetch_L1, prefetch_L2 | Insert `prefetch.global.L1/L2` after param loads, grouped per 128-byte cache line |
| StoreCacheHintTransform | `transform/store_cache_hints.py` | st_cache_cs, st_cache_wt, st_cache_wb | Add cache policies (write-back/write-through/streaming) to `st.global` instructions |
| SplitVectorLoadsTransform | `transform/split_vectors.py` | split_ld | Reverse vectorize: expand `ld.global.v2/v4.f32` back to scalar loads |

Additionally, 2 new reorder strategies were added to the existing ReorderTransform:
- `reorder_lf` (loads_first): prioritize LSU load instructions
- `reorder_sl` (stores_last): push stores to end of schedule

And 3 existing parameters were added to the search space:
- `cache_ca`, `cache_cv` (previously excluded from greedy v1)
- `maxnreg_255` (previously excluded from greedy v1)

### Full Action Space (20 transforms)

| # | Action | Class | New? |
|---|--------|-------|------|
| 1 | vec_ld | VectorizeLoadsTransform | No |
| 2 | vec_st | VectorizeStoresTransform | **Yes** |
| 3 | cache_cs | CacheHintTransform | No |
| 4 | cache_cg | CacheHintTransform | No |
| 5 | cache_ca | CacheHintTransform | Added to search |
| 6 | cache_cv | CacheHintTransform | Added to search |
| 7 | maxnreg_32 | RegisterBudgetTransform | No |
| 8 | maxnreg_64 | RegisterBudgetTransform | No |
| 9 | maxnreg_128 | RegisterBudgetTransform | No |
| 10 | maxnreg_255 | RegisterBudgetTransform | Added to search |
| 11 | reorder_cp | ReorderTransform | No |
| 12 | reorder_il | ReorderTransform | No |
| 13 | reorder_lf | ReorderTransform | **Yes** |
| 14 | reorder_sl | ReorderTransform | **Yes** |
| 15 | prefetch_L1 | PrefetchTransform | **Yes** |
| 16 | prefetch_L2 | PrefetchTransform | **Yes** |
| 17 | st_cache_cs | StoreCacheHintTransform | **Yes** |
| 18 | st_cache_wt | StoreCacheHintTransform | **Yes** |
| 19 | st_cache_wb | StoreCacheHintTransform | **Yes** |
| 20 | split_ld | SplitVectorLoadsTransform | **Yes** |

### Independent Benchmark (20 transforms × 9 kernels)

**Script**: `exp-assembly/scripts/benchmark_transforms_v2.py`
**Data**: `exp-assembly/data/transform_benchmark_v2.json`

Each transform applied independently to 9 representative kernels, compiled, verified correct, measured:

| Transform | Mean Delta | Status | Notes |
|-----------|-----------|--------|-------|
| vec_ld | **-9.4%** | OK 9/9 | Known, validated |
| vec_st | **-10.2%** | OK 9/9 | **Best new transform** |
| maxnreg_128 | -8.3% | OK 9/9 | Known, validated |
| maxnreg_255 | -9.6% | OK 9/9 | Known, validated |
| maxnreg_64 | +18.7% | OK 9/9 | Hurts on large kernels |
| maxnreg_32 | +128.5% | OK 9/9 | Hurts on all kernels |
| cache_cs | +0.2% | OK 9/9 | Neutral |
| cache_cg | +10.7% | OK 9/9 | Penalty |
| cache_ca | +0.2% | OK 9/9 | Neutral |
| cache_cv | +10.7% | OK 9/9 | Same as cg |
| reorder_cp | +8.4% | OK 9/9 | Mixed across sizes |
| reorder_il | +5.8% | OK 9/9 | Mixed across sizes |
| reorder_lf | +8.4% | OK 9/9 | **Identical to cp** on gemm_tile (DAG too linear) |
| reorder_sl | +8.4% | OK 9/9 | **Identical to cp** on gemm_tile (DAG too linear) |
| prefetch_L1 | +0.3% | OK 9/9 | Neutral |
| prefetch_L2 | +0.3% | OK 9/9 | Neutral |
| st_cache_cs | +0.2% | OK 9/9 | Neutral |
| st_cache_wt | +0.2% | OK 9/9 | Neutral |
| st_cache_wb | +0.2% | OK 9/9 | Neutral |
| split_ld | n/a | NoChg 9/9 | Only applies after vectorize (no vector loads in baseline) |

**Findings from independent benchmark:**
1. **vec_st is the standout new transform** (-10.2% mean, comparable to vec_ld at -9.4%)
2. Prefetch instructions are neutral on single-thread kernels (no overlapping warps to benefit from prefetch)
3. Store cache hints are neutral (stores are infrequent relative to loads in gemm_tile)
4. New reorder strategies (loads_first, stores_last) produce identical schedules to critical_path on gemm_tile — the DAG is too linear to differentiate heuristics. More diverse kernels needed.
5. split_ld produces no change on unvectorized baseline (by design — it reverses vectorize)

### Greedy Search v2 (20 transforms × 64 kernels)

**Script**: `exp-assembly/scripts/greedy_search_v2.py`
**Data**: `exp-assembly/data/greedy_search_v2_results.json`

Sequential greedy search with conflict groups (e.g., only one cache policy per transform class). Max 6 steps per kernel.

| Metric | v1 (9 actions) | v2 (20 actions) | Delta |
|--------|---------------|-----------------|-------|
| Mean improvement | -20.1% | **-28.1%** | **-8.1pp** |
| Median improvement | -15.1% | -27.1% | -12.0pp |
| Best | -50.4% | **-54.3%** | -3.9pp |
| Worst | +0.0% | +0.0% | 0pp |
| Wall time | ~15 min | 6030s (~100 min) | 6.7x (more actions to try) |

**Transform frequency in greedy v2 solutions:**

| Transform | Frequency | Role |
|-----------|-----------|------|
| vec_st | 56/64 (88%) | **Nearly universal — mirrors vec_ld** |
| vec_ld | 54/64 (84%) | Nearly universal |
| maxnreg_128 | 20/64 (31%) | Dominant register budget |
| reorder_cp | 14/64 (22%) | Now appears in composition (never appeared in v1) |
| maxnreg_64 | 9/64 (14%) | For smaller kernels |
| reorder_il | 9/64 (14%) | In composition |
| maxnreg_255 | 9/64 (14%) | For larger kernels |
| cache_cg | 7/64 (11%) | In composition after vec |
| cache_cs | 5/64 (8%) | Mild safe addition |
| reorder_sl | 3/64 (5%) | Rare |
| st_cache_wt | ~3/64 | In composition |
| st_cache_cs | ~2/64 | In composition |
| st_cache_wb | ~1/64 | Rare |
| prefetch_L1/L2 | 0/64 | Never selected |
| split_ld | 0/64 | Never applicable on baseline |

**Key v2 insights:**

1. **vectorize_stores is the primary source of the v2 improvement.** The -8.1pp mean gain mostly comes from vec_st, which appears in 88% of solutions and contributes -10.2% standalone. This was an overlooked optimization — v1 only vectorized loads, not stores.

2. **reorder now appears in greedy.** reorder_cp appears in 22% of v2 solutions (was 0% in v1). With more transforms available, the greedy search finds sequences where reorder adds value after vectorize + register budget. This validates that the v1 finding "reorder never selected" was an artifact of the limited action space.

3. **Store cache hints appear in composition.** st_cache_wt/cs/wb appear in ~5/64 solutions despite being neutral standalone. In specific transform sequences they provide marginal gains.

4. **Prefetch never selected.** Single-thread kernels don't benefit from prefetch (no concurrent warps to overlap with). Prefetch may matter for multi-warp production kernels.

5. **The best kernel (gemm_tile 4,6,8) improved from -50.4% to -54.3%.** The v2 sequence uses 4 steps: maxnreg_128 → reorder_cp → vec_ld → vec_st. In v1, the sequence was maxnreg_128 → vectorize (2 steps). The additional granularity of separate vec_ld/vec_st and the inclusion of reorder squeezed out 3.9pp more.

### B1 Status Summary

| Aspect | Before B1 | After B1 |
|--------|-----------|----------|
| Transform classes | 4 | 8 |
| Actions in search | 9 | 20 |
| Greedy mean | -20.1% | **-28.1%** |
| Greedy best | -50.4% | **-54.3%** |
| Greedy worst | +0.0% | +0.0% |
| Reorder in greedy | 0% | 22% |
| Data files | greedy_search_results.json | + greedy_search_v2_results.json, transform_benchmark_v2.json |

### What B1 Means for Chronos v3

1. **Action space expansion works.** Going from 9 to 20 actions improved mean by 8.1pp. Diminishing returns are likely for more L2 transforms (the neutral transforms contribute little), but L3 transforms (loop restructuring, shared memory) could provide another step change.

2. **vectorize_stores was hiding in plain sight.** A mirror of the most impactful v1 transform (vectorize_loads) was never implemented. Systematically reviewing all instruction types for optimization opportunities (vectorize, cache hints, reorder) catches these.

3. **The RLVR training data is now richer.** v2 greedy trajectories average ~3 steps with 20 possible actions (vs ~2 steps with 9 actions in v1). More diverse sequences = better training signal for behavior cloning and RL.

4. **The ceiling of L2 transforms is becoming visible.** 2 kernels still show 0% improvement (gemm_tile 2,2,2 and 2,8,8). These have very few instructions or already hit register ceilings. Breaking past -54.3% requires L3+ transforms.

---

## References

### Papers
- AlphaEvolve: A coding agent for scientific and algorithmic discovery (Google DeepMind, 2025) — https://arxiv.org/abs/2506.13131
- CuAsmRL: Optimizing GPU SASS Schedules via Deep RL (CGO 2025) — 9-26% at SASS level
- CompilerDream: DreamerV3 world model for LLVM pass ordering
- PEAK: Performance Engineering AI-Assistant for GPU Kernels — https://arxiv.org/abs/2512.19018
- The Bitter Lesson (Sutton, 2019) — http://www.incompleteideas.net/IncIdeas/BitterLesson.html
- OaK Architecture (Sutton, RLC 2025) — https://www.amii.ca/videos/oak-architecture-rich-sutton-rlc2025
- Loss of Plasticity in Deep Continual Learning (Nature 2024) — https://www.nature.com/articles/s41586-024-07711-7

### Existing Systems
- AlphaEvolve / OpenEvolve: https://github.com/jamesahou/openevolve
- CodeEvolve: https://arxiv.org/html/2510.14150v1
- Harmonic (formal verification for AI reasoning): https://harmonic.fun/
- Triton compiler: https://github.com/triton-lang/triton

### Chronos Data
- CHRONOS_V2_EXPERIMENT_LOG.md — complete v2 experiment results
- chat-with-aaditya.txt — conversation establishing the v3 direction
- All code under experiments/chronos/ (see codebase map in v2 log)
- exp-assembly/data/greedy_search_results.json — v1 greedy search (9 actions, 64 kernels)
- exp-assembly/data/greedy_search_v2_results.json — v2 greedy search (20 actions, 64 kernels)
- exp-assembly/data/transform_benchmark_v2.json — independent benchmark (20 transforms × 9 kernels)
