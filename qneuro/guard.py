"""Sole authority for quantum-advantage claims — no other module may phrase one."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from qneuro.types import FoldScores, GuardVerdict, VerdictCategory
from qneuro.models.schemas import BenchmarkRun, GuardVerdict as LegacyGuardVerdict

MIN_FOLDS_REQUIRED = 5
ADVANTAGE_THRESHOLD = 0.02


def nadeau_bengio_t_test(
    diffs: np.ndarray | list[float],
    test_train_ratio: float = 0.25,
) -> tuple[float, float]:
    """Compute Nadeau-Bengio corrected resampled t-test statistic and two-tailed p-value.

    Corrects for fold non-independence in repeated cross-validation:
    sigma^2_corrected = (1/J + n_te / n_tr) * sigma^2_diffs.
    For standard 5-fold CV, n_te / n_tr = 0.25.
    """
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    if j < 2:
        return 0.0, 1.0

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d == 0.0:
        # Exactly zero variance in fold differences
        return 0.0, 1.0 if mean_d == 0.0 else 0.0

    corrected_var = (1.0 / j + test_train_ratio) * var_d
    se_nb = float(np.sqrt(max(0.0, corrected_var)))
    if se_nb == 0.0:
        return 0.0, 1.0

    t_stat = mean_d / se_nb
    df = j - 1
    p_value = 2.0 * float(1.0 - stats.t.cdf(abs(t_stat), df=df))
    return float(t_stat), float(np.clip(p_value, 0.0, 1.0))


def bootstrap_ci(
    diffs: np.ndarray | list[float],
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval of the mean difference."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return 0.0, 0.0
    if len(d) == 1:
        return float(d[0]), float(d[0])

    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(d), size=(n_resamples, len(d)))
    resampled_means = np.mean(d[indices], axis=1)

    alpha = 1.0 - confidence_level
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_low = float(np.percentile(resampled_means, low_pct))
    ci_high = float(np.percentile(resampled_means, high_pct))
    return ci_low, ci_high


def holm_correction(p_values: list[float]) -> list[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values.

    Controls family-wise error rate across m hypothesis tests.
    """
    m = len(p_values)
    if m <= 1:
        return [float(np.clip(p, 0.0, 1.0)) for p in p_values]

    # Sort indices by unadjusted p-value
    sorted_indices = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m

    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        mult = m - rank
        raw_p = p_values[idx]
        adj = min(1.0, raw_p * mult)
        cum_max = max(cum_max, adj)
        adjusted[idx] = min(1.0, cum_max)

    return adjusted


def evaluate_advantage(
    quantum_scores: Sequence[FoldScores],
    classical_scores: Sequence[FoldScores],
    metric: str = "auroc",
    test_train_ratio: float = 0.25,
) -> GuardVerdict:
    """Sole function authorized to formulate and decide a quantum advantage claim.

    Methodology:
    1. Filter scores to designated metric (AUROC preferred).
    2. Identify the best classical baseline by mean score.
    3. Evaluate every quantum configuration against the best classical baseline across paired fold scores.
    4. Apply the Nadeau-Bengio corrected resampled t-test to each quantum candidate.
    5. Apply Holm step-down adjustment across all quantum candidates tried.
    6. Derive bootstrap 95% CI on the difference of the winning quantum config vs best classical.
    7. Issue verdict with conservative delta thresholds and corrected p-values.
    """
    q_filtered = [s for s in quantum_scores if s.metric == metric]
    c_filtered = [s for s in classical_scores if s.metric == metric]

    if not q_filtered or not c_filtered:
        return GuardVerdict(
            category=VerdictCategory.INSUFFICIENT_DATA,
            metric_used=metric,
            best_quantum_model="None",
            best_classical_model="None",
            quantum_score=0.0,
            classical_score=0.0,
            delta=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            configs_tried=len(q_filtered),
            sentence="Insufficient metric data available to conduct model arbitration.",
        )

    # Best classical baseline
    best_c = max(c_filtered, key=lambda s: s.mean_score)
    c_folds = np.asarray(best_c.scores, dtype=float)

    if len(c_folds) < MIN_FOLDS_REQUIRED:
        return GuardVerdict(
            category=VerdictCategory.INSUFFICIENT_DATA,
            metric_used=metric,
            best_quantum_model=q_filtered[0].config_id,
            best_classical_model=best_c.config_id,
            quantum_score=float(q_filtered[0].mean_score),
            classical_score=float(best_c.mean_score),
            delta=float(q_filtered[0].mean_score - best_c.mean_score),
            ci_lower=0.0,
            ci_upper=0.0,
            p_value=1.0,
            configs_tried=len(q_filtered),
            sentence=f"Insufficient fold count ({len(c_folds)} < {MIN_FOLDS_REQUIRED}) to establish statistical validity.",
        )

    # Test all quantum candidates against the best classical baseline
    raw_p_values: list[float] = []
    diffs_list: list[np.ndarray] = []

    for q in q_filtered:
        q_folds = np.asarray(q.scores, dtype=float)
        if len(q_folds) != len(c_folds):
            # Mismatched fold counts
            diffs = np.zeros(len(c_folds))
            raw_p = 1.0
        else:
            diffs = q_folds - c_folds
            _, raw_p = nadeau_bengio_t_test(diffs, test_train_ratio=test_train_ratio)
        diffs_list.append(diffs)
        raw_p_values.append(raw_p)

    corrected_p_values = holm_correction(raw_p_values)

    # Select best quantum model by mean score
    best_q_idx = int(np.argmax([q.mean_score for q in q_filtered]))
    best_q = q_filtered[best_q_idx]
    q_score = float(best_q.mean_score)
    c_score = float(best_c.mean_score)
    delta = float(q_score - c_score)

    best_diffs = diffs_list[best_q_idx]
    ci_low, ci_high = bootstrap_ci(best_diffs)
    best_corr_p = float(corrected_p_values[best_q_idx])

    configs_tried = len(q_filtered)
    metric_label = metric.upper()

    # Verdict categorization
    if delta > ADVANTAGE_THRESHOLD and best_corr_p < 0.05 and ci_low > 0.0:
        category = VerdictCategory.ADVANTAGE_SUPPORTED
        sentence = (
            f"Quantum configuration {best_q.config_id} demonstrated a statistically supported advantage "
            f"over classical baseline {best_c.config_id} with a {metric_label} delta of {delta:+.4f} "
            f"(bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], Nadeau-Bengio corrected p = {best_corr_p:.4f} "
            f"adjusted across {configs_tried} quantum configurations)."
        )
    elif delta < -ADVANTAGE_THRESHOLD and best_corr_p < 0.05 and ci_high < 0.0:
        category = VerdictCategory.CLASSICAL_BETTER
        sentence = (
            f"Classical baseline {best_c.config_id} outperformed best quantum configuration {best_q.config_id} "
            f"by {abs(delta):.4f} {metric_label} ({c_score:.4f} vs {q_score:.4f}, bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], "
            f"Nadeau-Bengio corrected p = {best_corr_p:.4f} across {configs_tried} quantum configurations). "
            f"No quantum advantage observed."
        )
    else:
        category = VerdictCategory.NO_DIFFERENCE
        sentence = (
            f"Observed {metric_label} delta of {delta:+.4f} between {best_q.config_id} ({q_score:.4f}) and "
            f"{best_c.config_id} ({c_score:.4f}) with bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}] "
            f"and Nadeau-Bengio corrected p = {best_corr_p:.4f} across {configs_tried} quantum configurations "
            f"does not reject the null hypothesis at threshold {ADVANTAGE_THRESHOLD:.2f}. "
            f"No conclusive advantage demonstrated."
        )

    return GuardVerdict(
        category=category,
        metric_used=metric,
        best_quantum_model=best_q.config_id,
        best_classical_model=best_c.config_id,
        quantum_score=q_score,
        classical_score=c_score,
        delta=delta,
        ci_lower=ci_low,
        ci_upper=ci_high,
        p_value=best_corr_p,
        configs_tried=configs_tried,
        sentence=sentence,
    )


def evaluate(run: BenchmarkRun, metric: str = "roc_auc") -> LegacyGuardVerdict:
    """Legacy compatibility bridge for BenchmarkRun schema objects."""
    threshold = 0.02
    classical_has_metric = any(getattr(r, metric, None) is not None for r in run.classical_results)
    quantum_has_metric = any(getattr(r, metric, None) is not None for r in run.quantum_results)
    if not (classical_has_metric and quantum_has_metric):
        metric = "f1_score"

    best_classical = max(run.classical_results, key=lambda r: getattr(r, metric) or 0.0)
    best_quantum = max(run.quantum_results, key=lambda r: getattr(r, metric) or 0.0)

    classical_best = float(getattr(best_classical, metric) or 0.0)
    quantum_best = float(getattr(best_quantum, metric) or 0.0)
    delta = float(quantum_best - classical_best)

    q_folds = getattr(best_quantum, f"{metric}_folds", None) or best_quantum.roc_auc_folds
    c_folds = getattr(best_classical, f"{metric}_folds", None) or best_classical.roc_auc_folds

    gap_ci_low = delta
    gap_ci_high = delta
    p_val = 1.0
    p_val_corr = 1.0

    if q_folds and c_folds and len(q_folds) == len(c_folds) and len(q_folds) >= 2:
        diffs = np.array(q_folds, dtype=float) - np.array(c_folds, dtype=float)
        k = len(diffs)
        std_diff = float(np.std(diffs, ddof=1))
        se_diff = std_diff / np.sqrt(k)
        t_crit = float(stats.t.ppf(0.975, df=k - 1))
        gap_ci_low = float(delta - t_crit * se_diff)
        gap_ci_high = float(delta + t_crit * se_diff)
        try:
            res = stats.ttest_rel(q_folds, c_folds)
            computed_p = float(res.pvalue)
            p_val = computed_p if not np.isnan(computed_p) else 1.0
            p_val_corr = min(1.0, p_val * max(1, len(run.classical_results)))
        except Exception:
            p_val = 1.0
            p_val_corr = 1.0

    winning_model = best_quantum.model_name if delta > 0 else best_classical.model_name
    metric_label = "AUROC" if metric == "roc_auc" else metric.upper()

    if delta > threshold:
        quantum_better = True
        claim = (
            f"{best_quantum.model_name} led classical {best_classical.model_name} "
            f"by {delta:+.4f} {metric_label} (95% CI [{gap_ci_low:+.4f}, {gap_ci_high:+.4f}], "
            f"corrected p = {p_val_corr:.4f}) on {run.n_samples} samples; "
            f"this exploratory signal on small cohorts should not be generalised without larger replication studies."
        )
    elif delta < -threshold:
        quantum_better = False
        claim = (
            f"Classical {best_classical.model_name} outperformed {best_quantum.model_name} "
            f"by {abs(delta):.4f} {metric_label} ({classical_best:.4f} vs {quantum_best:.4f}, "
            f"95% CI [{gap_ci_low:+.4f}, {gap_ci_high:+.4f}], corrected p = {p_val_corr:.4f}). "
            f"No quantum advantage observed."
        )
    else:
        quantum_better = False
        claim = (
            f"Observed {metric_label} delta of {delta:+.4f} ({best_quantum.model_name} {quantum_best:.4f} vs "
            f"{best_classical.model_name} {classical_best:.4f}, 95% CI [{gap_ci_low:+.4f}, {gap_ci_high:+.4f}], "
            f"corrected p = {p_val_corr:.4f}) remains within the noise margin (threshold = {threshold:.2f}). "
            f"No conclusive advantage demonstrated for either method."
        )

    return LegacyGuardVerdict(
        run_id=run.run_id,
        claim=claim,
        quantum_better=quantum_better,
        metric_used=metric,
        winning_model=winning_model,
        best_classical_model=best_classical.model_name,
        best_quantum_model=best_quantum.model_name,
        classical_best=classical_best,
        quantum_best=quantum_best,
        delta=delta,
        gap_ci_lower=gap_ci_low,
        gap_ci_upper=gap_ci_high,
        p_value=p_val,
        p_value_corrected=p_val_corr,
        confidence_note=f"Based on {run.n_samples} samples across 5 stratified folds with seed {run.random_seed}.",
    )

