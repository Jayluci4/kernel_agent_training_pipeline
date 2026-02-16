"""BC pretraining with Cross-Attention architecture.

Uses the new Query-Key ChronosNet:
  H (Keys) = GNN(node_features) — instruction embeddings
  Q (Query) = MLP(pipeline_state) — hardware context
  Value = multi-head cross-attention(Q, H) → MLP → scalar

The schedule_position feature (node_features[:, -1]) makes H encode
each instruction's position in the schedule. Q attends to H from
multiple perspectives to predict cycle count.

Also trains step-by-step policy imitation on the top-K schedules.
The policy uses Q · H^T dot-product to select next instruction.

Usage:
    python -m pipeline.training.train_bc_crossattn
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.model.representation import RepresentationNetwork, LatentState
from pipeline.model.prediction import PredictionNetwork
from pipeline.env.schedule_env import ScheduleEnv
from pipeline.env.instruction import Instruction, parse_ptx_body
from pipeline.harness.ptx_templates import gemm_tile


def load_dataset(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_obs(obs: Dict, device: torch.device,
                augment_edges: bool = False) -> Dict[str, torch.Tensor]:
    """Convert numpy observation to GPU tensors.

    If augment_edges=True, appends scheduling distance to edge features.
    For each DAG edge (src -> dst), computes:
      step_distance = schedule_position[dst] - schedule_position[src]
    This gives the GNN direct access to how far apart dependent
    instructions are in the schedule — the key signal for predicting stalls.
    """
    result = {}
    for key, arr in obs.items():
        if key == 'ready_mask':
            result[key] = torch.tensor(arr, dtype=torch.bool, device=device)
        elif key == 'edge_index':
            result[key] = torch.tensor(arr, dtype=torch.long, device=device)
        else:
            result[key] = torch.tensor(arr, dtype=torch.float32, device=device)
    nf = result['node_features']
    # Scheduled flag is at position 40 in node_features
    # (after 30 opcode + 7 pipeline + 1 latency + 1 depth + 1 height)
    SCHEDULED_IDX = 40
    if nf.shape[1] > SCHEDULED_IDX:
        result['scheduled_mask'] = nf[:, SCHEDULED_IDX] > 0.5
    else:
        result['scheduled_mask'] = torch.ones(nf.shape[0], dtype=torch.bool, device=device)

    if augment_edges:
        edge_index = result['edge_index']  # [2, E]
        edge_attr = result['edge_attr']    # [E, 3]
        if edge_index.shape[1] > 0:
            # schedule_position is the last node feature (appended by gen_bc_data)
            schedule_pos = nf[:, -1]  # [N], 0-1 normalized step position
            src, dst = edge_index[0], edge_index[1]
            step_distance = schedule_pos[dst] - schedule_pos[src]  # [E]
            edge_attr = torch.cat([edge_attr, step_distance.unsqueeze(1)], dim=1)
            result['edge_attr'] = edge_attr

    return result


def evaluate_value_spearman(
    rep_net: RepresentationNetwork,
    pred_net: PredictionNetwork,
    observations: List[Dict],
    hw_cycles: np.ndarray,
    device: torch.device,
    baseline: float = 0.0,
    scale: float = 1.0,
    augment_edges: bool = False,
) -> Tuple[float, float]:
    """Evaluate Spearman of value predictions vs actual cycles."""
    rep_net.eval()
    pred_net.eval()
    preds = []
    with torch.no_grad():
        for obs in observations:
            obs_t = prepare_obs(obs, device, augment_edges=augment_edges)
            state = rep_net(
                obs_t['node_features'], obs_t['edge_index'],
                obs_t['edge_attr'], obs_t['pipeline_state'],
                obs_t['register_pressure'], obs_t['ready_mask'],
                obs_t['scheduled_mask'],
            )
            _, value = pred_net(state)
            preds.append(value.item())
    preds = np.array(preds)
    targets = (baseline - hw_cycles) / scale
    sp, _ = spearmanr(preds, -hw_cycles)
    mse = float(np.mean((preds - targets) ** 2))
    return sp, mse


def train_value_epoch(
    rep_net: RepresentationNetwork,
    pred_net: PredictionNetwork,
    optimizer: optim.Optimizer,
    train_obs: List[Dict],
    train_cycles: np.ndarray,
    device: torch.device,
    baseline: float = 0.0,
    scale: float = 1.0,
    batch_size: int = 64,
    ranking_weight: float = 5.0,
    margin: float = 0.1,
    augment_edges: bool = False,
) -> Dict:
    """One epoch of value training with MSE + ranking loss."""
    rep_net.train()
    pred_net.train()
    n = len(train_obs)
    indices = np.random.permutation(n)
    ranking_fn = nn.MarginRankingLoss(margin=margin)

    total_mse = 0.0
    total_rank = 0.0
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        if len(batch_idx) < 2:
            continue

        values = []
        targets = []
        for i in batch_idx:
            obs_t = prepare_obs(train_obs[i], device,
                                augment_edges=augment_edges)
            state = rep_net(
                obs_t['node_features'], obs_t['edge_index'],
                obs_t['edge_attr'], obs_t['pipeline_state'],
                obs_t['register_pressure'], obs_t['ready_mask'],
                obs_t['scheduled_mask'],
            )
            _, value = pred_net(state)
            values.append(value)
            targets.append((baseline - train_cycles[i]) / scale)

        pred = torch.stack(values)
        target = torch.tensor(targets, dtype=torch.float32, device=device)

        mse_loss = nn.functional.mse_loss(pred, target)

        rank_loss = torch.tensor(0.0, device=device)
        B = len(pred)
        if B >= 2:
            idx_i = torch.arange(B, device=device).unsqueeze(1).expand(-1, B)
            idx_j = torch.arange(B, device=device).unsqueeze(0).expand(B, -1)
            mask = idx_i < idx_j
            pi, pj = pred[idx_i[mask]], pred[idx_j[mask]]
            ti, tj = target[idx_i[mask]], target[idx_j[mask]]
            y = torch.sign(ti - tj)
            non_tie = y != 0
            if non_tie.any():
                rank_loss = ranking_fn(pi[non_tie], pj[non_tie], y[non_tie])

        loss = mse_loss + ranking_weight * rank_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(rep_net.parameters()) + list(pred_net.parameters()),
            max_norm=1.0)
        optimizer.step()

        total_mse += mse_loss.item()
        total_rank += rank_loss.item()
        total_loss += loss.item()
        n_batches += 1

    return {
        'mse': total_mse / max(n_batches, 1),
        'rank_loss': total_rank / max(n_batches, 1),
        'total_loss': total_loss / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--edge-features', action='store_true',
                        help='Augment edges with scheduling distance')
    parser.add_argument('--ranking-weight', type=float, default=5.0)
    parser.add_argument('--checkpoint-name', type=str, default='best_crossattn.pt')
    args = parser.parse_args()

    print("=" * 60)
    print("BC Pretraining: Cross-Attention (Query-Key) Architecture")
    print("=" * 60)

    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    dataset_path = args.dataset or os.path.join(data_dir, 'bc_dataset.pkl')
    dataset = load_dataset(dataset_path)

    observations = dataset['observations']
    hw_cycles = np.array(dataset['hw_cycles'], dtype=np.float32)
    n = len(observations)
    print(f"Loaded {n} samples from {dataset_path}")
    print(f"Cycle range: {hw_cycles.min():.0f} - {hw_cycles.max():.0f} "
          f"(mean={hw_cycles.mean():.1f}, std={hw_cycles.std():.1f})")

    # Train/val split
    n_train = int(n * 0.8)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_obs = [observations[i] for i in train_idx]
    train_cycles = hw_cycles[train_idx]
    val_obs = [observations[i] for i in val_idx]
    val_cycles = hw_cycles[val_idx]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Compute normalization from training data (kernel-agnostic)
    baseline = float(np.median(train_cycles))
    q75, q25 = np.percentile(train_cycles, 75), np.percentile(train_cycles, 25)
    scale = max(float(q75 - q25), 1.0)
    print(f"Normalization: baseline={baseline:.1f} (median), scale={scale:.1f} (IQR)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    node_feat_dim = observations[0]['node_features'].shape[1]
    hidden_dim = 128

    # Edge feature dim: 3 base (RAW/WAR/WAW) + 1 scheduling distance if enabled
    base_edge_dim = observations[0]['edge_attr'].shape[1] if len(observations[0]['edge_attr']) > 0 else 3
    edge_feat_dim = base_edge_dim + (1 if args.edge_features else 0)
    print(f"Edge features: {'ON' if args.edge_features else 'OFF'} "
          f"(dim={edge_feat_dim})")

    rep_net = RepresentationNetwork(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        num_gat_layers=3,
        num_heads=4,
        use_edge_features=args.edge_features,
    ).to(device)

    pred_net = PredictionNetwork(
        hidden_dim=hidden_dim,
        num_value_heads=4,
        value_layers=2,
    ).to(device)

    total_params = sum(p.numel() for p in rep_net.parameters()) + \
                   sum(p.numel() for p in pred_net.parameters())
    print(f"Parameters: {total_params:,} "
          f"(rep={sum(p.numel() for p in rep_net.parameters()):,}, "
          f"pred={sum(p.numel() for p in pred_net.parameters()):,})")

    # Evaluate before training
    sp0, mse0 = evaluate_value_spearman(
        rep_net, pred_net, val_obs, val_cycles, device,
        baseline=baseline, scale=scale,
        augment_edges=args.edge_features)
    print(f"\nEpoch 0: Spearman={sp0:.4f}, MSE={mse0:.4f}")

    # Training
    all_params = list(rep_net.parameters()) + list(pred_net.parameters())
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_sp = sp0
    best_epoch = 0
    no_improve = 0

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(patience={args.patience})...")
    print(f" {'Epoch':>5} {'Loss':>9} {'MSE':>9} {'Rank':>9} "
          f"{'ValSp':>9} {'LR':>12}")
    print("-" * 65)

    save_dir = os.path.join(data_dir, 'bc_checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        metrics = train_value_epoch(
            rep_net, pred_net, optimizer,
            train_obs, train_cycles, device,
            baseline=baseline, scale=scale,
            batch_size=args.batch_size,
            ranking_weight=args.ranking_weight,
            augment_edges=args.edge_features,
        )
        scheduler.step()

        val_sp, val_mse = evaluate_value_spearman(
            rep_net, pred_net, val_obs, val_cycles, device,
            baseline=baseline, scale=scale,
            augment_edges=args.edge_features)
        current_lr = scheduler.get_last_lr()[0]

        marker = ""
        if val_sp > best_sp:
            best_sp = val_sp
            best_epoch = epoch
            no_improve = 0
            marker = " *"
            torch.save({
                'epoch': epoch,
                'rep_state': rep_net.state_dict(),
                'pred_state': pred_net.state_dict(),
                'spearman': val_sp,
                'baseline': baseline,
                'scale': scale,
                'edge_features': args.edge_features,
            }, os.path.join(save_dir, args.checkpoint_name))
        else:
            no_improve += 1

        if epoch <= 20 or epoch % 10 == 0 or marker:
            print(f" {epoch:>5} {metrics['total_loss']:>9.4f} "
                  f"{metrics['mse']:>9.4f} {metrics['rank_loss']:>9.4f} "
                  f"{val_sp:>9.4f} {current_lr:>12.6f}{marker}")

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    print(f"\n{'='*60}")
    print(f"Best val Spearman: {best_sp:.4f} at epoch {best_epoch}")
    print(f"Pass criteria: Spearman > 0.8 -> "
          f"{'PASS' if best_sp > 0.8 else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
