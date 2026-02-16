#!/usr/bin/env python3
"""Behavior cloning from greedy search v2 trajectories.

Trains a gradient boosted tree to predict the next action from state features.
This is the warm-start baseline before RLVR training.

Evaluation:
  - Leave-one-kernel-out cross-validation
  - Per-action precision/recall
  - Comparison with "always vec_ld" and random baselines

Output: exp-assembly/data/bc_results.json

Usage:
    python3 exp-assembly/scripts/train_bc.py
"""

import sys
import os
import json

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from pipeline.features.kernel_features import FEATURE_NAMES

DATA_DIR = os.path.join(REPO_ROOT, "data")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_trajectories():
    """Load trajectory data from JSONL file."""
    path = os.path.join(DATA_DIR, "trajectories_v2.jsonl")
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def prepare_dataset(entries):
    """Convert trajectory entries to (X, y, kernel_ids) arrays."""
    X = []
    y = []
    kernel_ids = []

    for entry in entries:
        X.append(entry["feature_array"])
        y.append(entry["action_id"])
        kernel_ids.append(entry["kernel"])

    return np.array(X), np.array(y), kernel_ids


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def train_gbt(X_train, y_train, n_classes):
    """Train a GradientBoostingClassifier."""
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train, n_classes):
    """Train a RandomForestClassifier."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def leave_one_kernel_out(X, y, kernel_ids, train_fn, n_classes):
    """LOKO cross-validation: train on 63 kernels, test on 1."""
    unique_kernels = sorted(set(kernel_ids))
    all_preds = np.zeros_like(y)
    all_true = np.zeros_like(y)

    for held_out in unique_kernels:
        train_mask = np.array([k != held_out for k in kernel_ids])
        test_mask = ~train_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model = train_fn(X_train, y_train, n_classes)
        preds = model.predict(X_test)

        all_preds[test_mask] = preds
        all_true[test_mask] = y_test

    return all_true, all_preds


def evaluate_predictions(y_true, y_pred, action_names):
    """Compute accuracy and per-action metrics."""
    accuracy = np.mean(y_true == y_pred)

    # Per-action precision and recall
    per_action = {}
    for action_id, name in enumerate(action_names):
        true_mask = y_true == action_id
        pred_mask = y_pred == action_id

        n_true = true_mask.sum()
        n_pred = pred_mask.sum()
        n_correct = (true_mask & pred_mask).sum()

        if n_true == 0 and n_pred == 0:
            continue

        precision = n_correct / max(n_pred, 1)
        recall = n_correct / max(n_true, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_action[name] = {
            "n_true": int(n_true),
            "n_pred": int(n_pred),
            "n_correct": int(n_correct),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    return {
        "accuracy": round(accuracy, 4),
        "per_action": per_action,
    }


def majority_baseline(y, action_names):
    """Baseline: always predict the most common action."""
    from collections import Counter
    counts = Counter(y)
    majority_id = counts.most_common(1)[0][0]
    majority_name = action_names[majority_id]
    majority_acc = counts[majority_id] / len(y)
    return majority_name, majority_acc


def random_baseline(y, n_trials=1000):
    """Baseline: random action prediction."""
    n_classes = len(set(y))
    accs = []
    rng = np.random.RandomState(42)
    for _ in range(n_trials):
        preds = rng.randint(0, n_classes, size=len(y))
        accs.append(np.mean(preds == y))
    return np.mean(accs)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(X, y, n_classes):
    """Train on full data, return feature importances."""
    model = train_gbt(X, y, n_classes)
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    return ranked, model


# ---------------------------------------------------------------------------
# Greedy replay evaluation
# ---------------------------------------------------------------------------

def evaluate_greedy_replay(entries, model, action_names):
    """Simulate greedy search using BC model predictions.

    For each kernel, start from baseline features and repeatedly
    predict the next action until "stop" is predicted or max steps reached.
    Compare predicted sequences with actual greedy sequences.
    """
    # Group entries by kernel
    kernels = {}
    for entry in entries:
        kid = entry["kernel"]
        if kid not in kernels:
            kernels[kid] = []
        kernels[kid].append(entry)

    results = []
    exact_match = 0
    first_action_match = 0

    for kid, kernel_entries in sorted(kernels.items()):
        actual_actions = [e["action"] for e in kernel_entries]

        # BC model predicts from first state
        first_features = np.array(kernel_entries[0]["feature_array"]).reshape(1, -1)
        pred_action_id = model.predict(first_features)[0]
        pred_action = action_names[pred_action_id]

        if actual_actions[0] == pred_action:
            first_action_match += 1

        if actual_actions == [pred_action]:  # simplified
            exact_match += 1

        results.append({
            "kernel": kid,
            "actual_first": actual_actions[0],
            "predicted_first": pred_action,
            "match": actual_actions[0] == pred_action,
            "actual_sequence": actual_actions,
        })

    return {
        "n_kernels": len(kernels),
        "first_action_accuracy": round(first_action_match / max(len(kernels), 1), 4),
        "details": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading trajectories...")
    entries = load_trajectories()
    print(f"  Loaded {len(entries)} entries from {len(set(e['kernel'] for e in entries))} kernels")

    # Load action names from summary
    summary_path = os.path.join(DATA_DIR, "trajectory_summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    action_names = summary["action_names"]
    n_classes = len(action_names)

    X, y, kernel_ids = prepare_dataset(entries)
    print(f"  Features: {X.shape[1]}, Classes: {n_classes}, Samples: {len(y)}")

    # Baselines
    maj_name, maj_acc = majority_baseline(y, action_names)
    rand_acc = random_baseline(y)
    print(f"\n  Majority baseline ({maj_name}): {maj_acc:.1%}")
    print(f"  Random baseline: {rand_acc:.1%}")

    # LOKO evaluation
    print("\nRunning leave-one-kernel-out CV...")

    print("  GBT...")
    y_true_gbt, y_pred_gbt = leave_one_kernel_out(X, y, kernel_ids, train_gbt, n_classes)
    gbt_results = evaluate_predictions(y_true_gbt, y_pred_gbt, action_names)
    print(f"  GBT accuracy: {gbt_results['accuracy']:.1%}")

    print("  RF...")
    y_true_rf, y_pred_rf = leave_one_kernel_out(X, y, kernel_ids, train_rf, n_classes)
    rf_results = evaluate_predictions(y_true_rf, y_pred_rf, action_names)
    print(f"  RF accuracy: {rf_results['accuracy']:.1%}")

    # Feature importance (train on full data)
    print("\nFeature importance (GBT on full data):")
    ranked_features, full_model = get_feature_importance(X, y, n_classes)
    for name, imp in ranked_features[:10]:
        print(f"  {name}: {imp:.3f}")

    # Greedy replay with BC model
    print("\nGreedy replay evaluation (first action prediction):")
    replay_results = evaluate_greedy_replay(entries, full_model, action_names)
    print(f"  First action accuracy: {replay_results['first_action_accuracy']:.1%}")

    # Per-action breakdown
    print(f"\nPer-action metrics (GBT LOKO):")
    print(f"  {'Action':<16} {'True':>5} {'Pred':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"  {'-'*55}")
    for name, metrics in sorted(gbt_results["per_action"].items(),
                                 key=lambda x: -x[1]["n_true"]):
        print(f"  {name:<16} {metrics['n_true']:>5} {metrics['n_pred']:>5} "
              f"{metrics['precision']:>6.3f} {metrics['recall']:>6.3f} "
              f"{metrics['f1']:>6.3f}")

    # Save results
    output = {
        "n_samples": len(y),
        "n_features": X.shape[1],
        "n_classes": n_classes,
        "baselines": {
            "majority": {"action": maj_name, "accuracy": round(maj_acc, 4)},
            "random": {"accuracy": round(rand_acc, 4)},
        },
        "gbt_loko": gbt_results,
        "rf_loko": rf_results,
        "feature_importance": [
            {"feature": name, "importance": round(imp, 4)}
            for name, imp in ranked_features
        ],
        "replay": {
            "first_action_accuracy": replay_results["first_action_accuracy"],
        },
    }

    out_path = os.path.join(DATA_DIR, "bc_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
