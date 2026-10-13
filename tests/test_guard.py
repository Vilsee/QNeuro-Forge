"""Tests for the guard module — sole quantum-advantage arbiter."""
from __future__ import annotations

import datetime
import numpy as np
import pytest

from qneuro.guard import evaluate, evaluate_advantage, nadeau_bengio_t_test, bootstrap_ci, holm_correction
from qneuro.models.schemas import BenchmarkRun, ModelResult
from qneuro.types import FoldScores, VerdictCategory


def _make_fold_scores(config_id: str, scores: list[float], metric: str = "auroc") -> FoldScores:
    arr = np.array(scores, dtype=float)
    return FoldScores(
        config_id=config_id,
        metric=metric,
        scores=scores,
        mean_score=float(arr.mean()),
        std_score=float(arr.std(ddof=1)) if len(scores) > 1 else 0.0,
        ci_lower=float(arr.min()),
        ci_upper=float(arr.max()),
    )


def test_guard_identical_models_no_difference() -> None:
    """Identical fold performance between quantum and classical yields NO_DIFFERENCE."""
    folds = [0.65, 0.62, 0.68, 0.64, 0.66, 0.63, 0.67, 0.65, 0.64, 0.66]
    q_scores = [_make_fold_scores("q_model", folds)]
    c_scores = [_make_fold_scores("c_model", folds)]

    verdict = evaluate_advantage(q_scores, c_scores, metric="auroc")
    assert verdict.category == VerdictCategory.NO_DIFFERENCE
    assert verdict.delta == pytest.approx(0.0, abs=1e-6)
    assert verdict.p_value >= 0.05
    assert "does not reject the null hypothesis" in verdict.sentence


def test_guard_large_gap_advantage_supported() -> None:
    """A consistent, large positive gap above threshold yields ADVANTAGE_SUPPORTED."""
    c_folds = [0.55, 0.53, 0.56, 0.54, 0.55, 0.52, 0.57, 0.53, 0.54, 0.55]
    # Quantum is consistently +0.10 higher across every fold
    q_folds = [x + 0.10 for x in c_folds]

    q_scores = [_make_fold_scores("quantum_model", q_folds)]
    c_scores = [_make_fold_scores("classical_baseline", c_folds)]

    verdict = evaluate_advantage(q_scores, c_scores, metric="auroc")
    assert verdict.category == VerdictCategory.ADVANTAGE_SUPPORTED
    assert verdict.delta == pytest.approx(0.10, abs=1e-4)
    assert verdict.ci_lower > 0.0
    assert verdict.p_value < 0.05
    assert "statistically supported advantage" in verdict.sentence


def test_guard_classical_clearly_better() -> None:
    """When classical outperforms quantum consistently by a substantial margin."""
    c_folds = [0.80, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83, 0.80, 0.79, 0.81]
    # Quantum is 0.12 lower
    q_folds = [x - 0.12 for x in c_folds]

    q_scores = [_make_fold_scores("quantum_model", q_folds)]
    c_scores = [_make_fold_scores("classical_baseline", c_folds)]

    verdict = evaluate_advantage(q_scores, c_scores, metric="auroc")
    assert verdict.category == VerdictCategory.CLASSICAL_BETTER
    assert verdict.delta < -0.02
    assert verdict.ci_upper < 0.0
    assert verdict.p_value < 0.05
    assert "outperformed best quantum configuration" in verdict.sentence


def test_guard_insufficient_folds() -> None:
    """Fewer than minimum required folds (5) returns INSUFFICIENT_DATA."""
    q_scores = [_make_fold_scores("q_model", [0.7, 0.8, 0.75])]
    c_scores = [_make_fold_scores("c_model", [0.6, 0.65, 0.62])]

    verdict = evaluate_advantage(q_scores, c_scores, metric="auroc")
    assert verdict.category == VerdictCategory.INSUFFICIENT_DATA
    assert "Insufficient fold count" in verdict.sentence


def test_guard_simulation_false_claim_rate() -> None:
    """Run 200 simulation trials with 30 noise quantum configs against baseline.

    Verifies that the empirical false-positive rate (claiming ADVANTAGE_SUPPORTED
    when all candidates are null noise) remains well under or near alpha = 0.05
    thanks to Nadeau-Bengio correction and Holm family-wise correction.
    """
    rng = np.random.default_rng(2026)
    n_trials = 200
    n_quantum_configs = 30
    n_folds = 15  # 5 folds x 3 repeats
    false_claims = 0

    for trial in range(n_trials):
        # Baseline AUROC around 0.60 with typical fold variance
        base_folds = rng.normal(loc=0.60, scale=0.04, size=n_folds)
        c_scores = [_make_fold_scores("classical_baseline", list(base_folds))]

        # 30 quantum configs drawn from the EXACT SAME null distribution
        q_scores = []
        for i in range(n_quantum_configs):
            q_folds = rng.normal(loc=0.60, scale=0.04, size=n_folds)
            q_scores.append(_make_fold_scores(f"q_config_{i}", list(q_folds)))

        verdict = evaluate_advantage(q_scores, c_scores, metric="auroc")
        if verdict.category == VerdictCategory.ADVANTAGE_SUPPORTED:
            false_claims += 1

    false_claim_rate = false_claims / n_trials
    print(f"\nSimulation results: {false_claims}/{n_trials} false claims (rate = {false_claim_rate:.4f})")
    # Under Holm-Bonferroni correction and Nadeau-Bengio variance correction,
    # the false discovery rate under the null must be strictly <= 0.05 (typically 0.0)
    assert false_claim_rate <= 0.05, f"False claim rate {false_claim_rate:.4f} exceeded alpha = 0.05"


def _make_run(classical_f1: float, quantum_f1: float, seed: int = 1) -> BenchmarkRun:
    return BenchmarkRun(
        run_id="test-guard",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        dataset_name="test",
        dataset_source="SYNTHETIC",
        n_samples=40,
        n_features=4,
        classical_results=[
            ModelResult(
                model_name="Baseline",
                accuracy=classical_f1,
                f1_score=classical_f1,
                precision=classical_f1,
                recall=classical_f1,
                train_time_seconds=0.01,
            )
        ],
        quantum_results=[
            ModelResult(
                model_name="QKernel",
                accuracy=quantum_f1,
                f1_score=quantum_f1,
                precision=quantum_f1,
                recall=quantum_f1,
                train_time_seconds=0.02,
            )
        ],
        random_seed=seed,
        python_version="3.11.0",
    )


def test_quantum_better_above_threshold() -> None:
    run = _make_run(classical_f1=0.70, quantum_f1=0.75)
    verdict = evaluate(run)
    assert verdict.quantum_better is True
    assert verdict.delta == pytest.approx(0.05, abs=1e-6)
    assert "should not be generalised" in verdict.claim


def test_classical_better() -> None:
    run = _make_run(classical_f1=0.80, quantum_f1=0.70)
    verdict = evaluate(run)
    assert verdict.quantum_better is False
    assert "outperformed" in verdict.claim


def test_within_noise_margin() -> None:
    run = _make_run(classical_f1=0.75, quantum_f1=0.76)
    verdict = evaluate(run)
    assert verdict.quantum_better is False
    assert "noise margin" in verdict.claim


def test_confidence_note_mentions_sample_size() -> None:
    run = _make_run(classical_f1=0.70, quantum_f1=0.75)
    verdict = evaluate(run)
    assert "40 samples" in verdict.confidence_note
