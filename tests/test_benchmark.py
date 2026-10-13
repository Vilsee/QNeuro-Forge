"""Full benchmark integration test on synthetic data — prints real results table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qneuro.baselines import run_full_benchmark
from qneuro.data import generate_synthetic, compute_fingerprint, COLUMN_NAMES, TARGET_COLUMN


def test_full_benchmark_synthetic() -> None:
    """Run the complete benchmark pipeline on the 120-sample SYNTHETIC cohort.

    Prints an unedited results table. All numbers originate from this execution.
    """
    df, source = generate_synthetic(n_samples=120, random_seed=42)
    assert source == "SYNTHETIC"

    X = df[COLUMN_NAMES].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()
    fingerprint = compute_fingerprint(df)

    fold_scores, wall_clocks = run_full_benchmark(
        X, y,
        random_seed=42,
        n_splits=5,
        n_repeats=3,
        data_fingerprint=fingerprint,
    )

    # Verify structural invariants
    config_ids = sorted(set(fs.config_id for fs in fold_scores))
    assert len(config_ids) >= 7, f"Expected at least 7 configs, got {len(config_ids)}"

    metrics = sorted(set(fs.metric for fs in fold_scores))
    assert "auroc" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics

    # All fold score arrays must have identical length (5 folds x 3 repeats = 15)
    expected_n_folds = 15
    for fs in fold_scores:
        assert len(fs.scores) == expected_n_folds, (
            f"{fs.config_id}/{fs.metric}: expected {expected_n_folds} fold scores, got {len(fs.scores)}"
        )

    # Print the real results table
    print("\n" + "=" * 90)
    print("QNEURO FORGE BENCHMARK — SYNTHETIC COHORT (120 samples, 8 features)")
    print("RepeatedStratifiedKFold: 5 folds x 3 repeats = 15 evaluations per model")
    print("=" * 90)

    # Build a summary for AUROC (primary metric)
    auroc_rows = [fs for fs in fold_scores if fs.metric == "auroc"]
    auroc_rows.sort(key=lambda fs: fs.mean_score, reverse=True)

    print(f"\n{'Model':<45} {'AUROC':>8} {'±std':>8} {'95% CI':>18} {'Time(s)':>9}")
    print("-" * 90)
    for fs in auroc_rows:
        ci_str = f"[{fs.ci_lower:.4f}, {fs.ci_upper:.4f}]" if fs.ci_lower is not None else "N/A"
        wt = wall_clocks.get(fs.config_id, 0.0)
        print(f"{fs.config_id:<45} {fs.mean_score:>8.4f} {fs.std_score:>8.4f} {ci_str:>18} {wt:>9.2f}")

    # F1 table
    f1_rows = [fs for fs in fold_scores if fs.metric == "f1"]
    f1_rows.sort(key=lambda fs: fs.mean_score, reverse=True)

    print(f"\n{'Model':<45} {'F1':>8} {'±std':>8} {'95% CI':>18}")
    print("-" * 90)
    for fs in f1_rows:
        ci_str = f"[{fs.ci_lower:.4f}, {fs.ci_upper:.4f}]" if fs.ci_lower is not None else "N/A"
        print(f"{fs.config_id:<45} {fs.mean_score:>8.4f} {fs.std_score:>8.4f} {ci_str:>18}")

    # Accuracy table
    acc_rows = [fs for fs in fold_scores if fs.metric == "accuracy"]
    acc_rows.sort(key=lambda fs: fs.mean_score, reverse=True)

    print(f"\n{'Model':<45} {'Accuracy':>8} {'±std':>8} {'95% CI':>18}")
    print("-" * 90)
    for fs in acc_rows:
        ci_str = f"[{fs.ci_lower:.4f}, {fs.ci_upper:.4f}]" if fs.ci_lower is not None else "N/A"
        print(f"{fs.config_id:<45} {fs.mean_score:>8.4f} {fs.std_score:>8.4f} {ci_str:>18}")

    print("\n" + "=" * 90)
    print(f"Wall-clock totals:")
    for cid, wt in sorted(wall_clocks.items(), key=lambda kv: kv[1]):
        print(f"  {cid:<45} {wt:.2f}s")
    print("=" * 90)
