"""Re-running the Morphos experiments. Nothing here is typed in: every number is computed."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

from qneuro.guard import evaluate_advantage
from qneuro.study.data import DATA, Split, load_split
from qneuro.study.kernels import fidelity_kernel, product_kernel, zz_states
from qneuro.types import FoldScores

warnings.filterwarnings("ignore")


def _xgb(seed: int = 42):
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=seed, eval_metric="logloss")


def _ci95(v: list[float]) -> list[float]:
    a = np.array(v)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return [round(float(a.mean() - h), 4), round(float(a.mean() + h), 4)]


def _metrics(y, p, s) -> dict[str, float]:
    return dict(
        mcc=matthews_corrcoef(y, p),
        accuracy=accuracy_score(y, p),
        f1=f1_score(y, p),
        precision=precision_score(y, p),
        recall=recall_score(y, p),
        roc_auc=roc_auc_score(y, s),
    )


def mcc_from_confusion(cm: list[list[int]]) -> float:
    (tn, fp), (fn, tp) = cm
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else float((tp * tn - fp * fn) / den)


# --------------------------------------------------------------------------- E6
def benchmark_e6(split: Split | None = None, scale: float = 0.3, seed: int = 42) -> dict[str, Any]:
    """True ZZ kernel and five classical models on identical folds (the E6 recipe)."""
    split = split or load_split("withtime")
    Xtr, ytr, Xte, yte = split.Xtr, split.ytr, split.Xte, split.yte
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, ytr))

    selection: dict[str, dict[str, float]] = {}
    for sc in (1.0, 0.3):
        S = zz_states(Xtr * sc)
        Kt = fidelity_kernel(S, S)
        m = [
            matthews_corrcoef(
                ytr[b], SVC(kernel="precomputed").fit(Kt[np.ix_(a, a)], ytr[a]).predict(Kt[np.ix_(b, a)])
            )
            for a, b in folds
        ]
        St = zz_states(Xte * sc)
        p = SVC(kernel="precomputed").fit(Kt, ytr).predict(fidelity_kernel(St, S))
        selection[str(sc)] = dict(
            cv_mcc=round(float(np.mean(m)), 4),
            holdout_mcc=round(float(matthews_corrcoef(yte, p)), 4),
            holdout_accuracy=float(accuracy_score(yte, p)),
        )

    S = zz_states(Xtr * scale)
    Kt = fidelity_kernel(S, S)
    St = zz_states(Xte * scale)
    Ke = fidelity_kernel(St, S)

    def evalmodel(make, quantum=False):
        rows = []
        for a, b in folds:
            if quantum:
                c = SVC(kernel="precomputed").fit(Kt[np.ix_(a, a)], ytr[a])
                Xb = Kt[np.ix_(b, a)]
            else:
                c = make().fit(Xtr[a], ytr[a])
                Xb = Xtr[b]
            s = c.decision_function(Xb) if quantum or not hasattr(c, "predict_proba") else c.predict_proba(Xb)[:, 1]
            rows.append(_metrics(ytr[b], c.predict(Xb), s))
        if quantum:
            c = SVC(kernel="precomputed").fit(Kt, ytr)
            Xh = Ke
        else:
            c = make().fit(Xtr, ytr)
            Xh = Xte
        s = c.decision_function(Xh) if quantum or not hasattr(c, "predict_proba") else c.predict_proba(Xh)[:, 1]
        hold = _metrics(yte, c.predict(Xh), s)
        pred = c.predict(Xh)
        g = lambda k: [r[k] for r in rows]  # noqa: E731
        cv = dict(
            mcc_mean=round(float(np.mean(g("mcc"))), 4),
            mcc_ci_95=_ci95(g("mcc")),
            mcc_folds=[round(x, 4) for x in g("mcc")],
            accuracy_mean=round(float(np.mean(g("accuracy"))), 4),
            f1_mean=round(float(np.mean(g("f1"))), 4),
            roc_auc_mean=round(float(np.mean(g("roc_auc"))), 4),
        )
        cm = confusion_matrix(yte, pred).tolist()
        return dict(cv_5fold=cv, holdout_test={k: float(v) for k, v in hold.items()}, holdout_confusion=cm), g("mcc")

    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
        "RandomForest": lambda: RandomForestClassifier(random_state=42),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=42),
        "SVM-RBF": lambda: SVC(kernel="rbf"),
        "XGBoost": lambda: _xgb(42),
    }
    perf: dict[str, Any] = {}
    mccs: dict[str, list[float]] = {}
    perf["QSVM (Quantum Kernel)"], mccs["q"] = evalmodel(None, True)
    for k, f in models.items():
        perf[k], mccs[k] = evalmodel(f)

    d = np.array(mccs["q"]) - np.array(mccs["RandomForest"])
    tstat, pv = stats.ttest_rel(mccs["q"], mccs["RandomForest"])
    h = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
    stat = dict(
        delta_mcc=round(float(d.mean()), 4),
        delta_ci_95=[round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)],
        paired_t_statistic=round(float(tstat), 4),
        p_value=round(float(pv), 4),
    )
    return dict(performance_summary=perf, statistical_comparison=stat, scale_selection=selection, scale_used=scale)


SCORE_TOLERANCE = 5e-4  # ROC-AUC depends on float ordering of probabilities, so library versions can move it a little


def compare_with_stored_e6(result: dict[str, Any], stored_path: Path | None = None) -> dict[str, Any]:
    """Diff a fresh E6 run against the stored JSON (timing excluded).

    Class-prediction metrics (MCC, accuracy, F1, precision, recall) and every fold score must be identical.
    Score-based ROC-AUC may differ by up to SCORE_TOLERANCE, because it depends on the order of floating-point
    probabilities, which can change between scikit-learn versions and platforms.
    """
    stored = json.loads((stored_path or DATA / "submissions" / "E6_local_zz_NOT_submitted.json").read_text("utf-8"))
    diffs: list[str] = []
    score_diffs: list[dict[str, Any]] = []
    for m, row in result["performance_summary"].items():
        sm = stored["performance_summary"][m]
        for k in ("mcc_mean", "mcc_ci_95", "mcc_folds", "accuracy_mean", "f1_mean"):
            if row["cv_5fold"][k] != sm["cv_5fold"][k]:
                diffs.append(f"{m} cv {k}")
        a, b = row["cv_5fold"]["roc_auc_mean"], sm["cv_5fold"]["roc_auc_mean"]
        if a != b:
            score_diffs.append(dict(field=f"{m} cv roc_auc_mean", ours=a, stored=b, diff=abs(a - b)))
        for k, v in row["holdout_test"].items():
            if k == "train_time_seconds":
                continue
            d = abs(v - sm["holdout_test"][k])
            if k == "roc_auc":
                if d > 1e-12:
                    score_diffs.append(dict(field=f"{m} holdout roc_auc", ours=v, stored=sm["holdout_test"][k], diff=d))
            elif d > 1e-9:
                diffs.append(f"{m} holdout {k}")
    for k, v in result["statistical_comparison"].items():
        if v != stored["statistical_comparison"][k]:
            diffs.append(f"stat {k}")
    max_score = max((x["diff"] for x in score_diffs), default=0.0)
    return dict(
        n_differences=len(diffs),
        differences=diffs,
        score_differences=score_diffs,
        max_score_diff=max_score,
        identical=len(diffs) == 0 and max_score <= SCORE_TOLERANCE,
        bit_identical=len(diffs) == 0 and not score_diffs,
    )


# --------------------------------------------------------------------------- E5
def reproduce_e5(split: Split | None = None, submitted_path: Path | None = None) -> dict[str, Any]:
    """Independent re-implementation of the submitted QSVM (angle-embedding fidelity kernel)."""
    split = split or load_split("withtime")
    sub = json.loads((submitted_path or DATA / "submissions" / "SUBMITTED_corrected.json").read_text("utf-8"))
    perf = sub["performance_summary"]
    Xtr, ytr, Xte, yte = split.Xtr, split.ytr, split.Xte, split.yte
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, ytr))

    Kt = product_kernel(Xtr, Xtr, 1.0)
    Ke = product_kernel(Xte, Xtr, 1.0)
    fold_mcc = []
    for a, b in folds:
        p = SVC(kernel="precomputed").fit(Kt[np.ix_(a, a)], ytr[a]).predict(Kt[np.ix_(b, a)])
        fold_mcc.append(float(matthews_corrcoef(ytr[b], p)))
    clf = SVC(kernel="precomputed").fit(Kt, ytr)
    pred = clf.predict(Ke)
    hold = _metrics(yte, pred, clf.decision_function(Ke))
    cm = confusion_matrix(yte, pred).tolist()

    sq = perf["QSVM (Quantum Kernel)"]
    hold_sub = sq["holdout_test"]
    hold_diff = {k: abs(hold[k] - hold_sub[k]) for k in ("mcc", "accuracy", "f1", "precision", "recall", "roc_auc")}
    fold_sub = sq["cv_5fold"]["mcc_folds"]
    fold_match = [abs(round(m, 4) - s) < 5e-4 for m, s in zip(fold_mcc, fold_sub)]

    # The same recipe with the ZZ (entangling) kernel at the same input range, for contrast.
    S, St = zz_states(Xtr), zz_states(Xte)
    p_zz = SVC(kernel="precomputed").fit(fidelity_kernel(S, S), ytr).predict(fidelity_kernel(St, S))
    zz_hold_mcc = float(matthews_corrcoef(yte, p_zz))

    # Classical rows of the submitted file: try the recipes that reproduce them.
    classical: dict[str, Any] = {}
    recipes = {
        "LogisticRegression": (lambda: LogisticRegression(max_iter=1000), 0.3),
        "SVM-RBF": (lambda: SVC(kernel="rbf"), 0.3),
        "RandomForest": (lambda: RandomForestClassifier(random_state=42), 0.3),
        "GradientBoosting": (lambda: GradientBoostingClassifier(random_state=42), 0.3),
        "XGBoost": (lambda: _xgb(42), 0.3),
    }
    for name, (mk, sc) in recipes.items():
        ms = [
            matthews_corrcoef(ytr[b], mk().fit(Xtr[a] * sc, ytr[a]).predict(Xtr[b] * sc)) for a, b in folds
        ]
        h = float(matthews_corrcoef(yte, mk().fit(Xtr * sc, ytr).predict(Xte * sc)))
        sm = perf[name]
        ok = abs(np.mean(ms) - sm["cv_5fold"]["mcc_mean"]) < 5e-4 and abs(h - sm["holdout_test"]["mcc"]) < 5e-4
        classical[name] = dict(
            recipe=f"{name}, inputs x {sc}, folds seed 42",
            reproduced_cv_mcc=round(float(np.mean(ms)), 4),
            submitted_cv_mcc=sm["cv_5fold"]["mcc_mean"],
            reproduced_holdout_mcc=round(h, 4),
            submitted_holdout_mcc=round(sm["holdout_test"]["mcc"], 4),
            reproduced=bool(ok),
        )
    # Same models at the full [0, 1] range, to show what the 0.3 shrink did to logistic regression.
    lr_full = [
        matthews_corrcoef(
            ytr[b], LogisticRegression(max_iter=1000).fit(Xtr[a], ytr[a]).predict(Xtr[b])
        )
        for a, b in folds
    ]
    lr_full_hold = float(
        matthews_corrcoef(yte, LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte))
    )

    # Paired comparison against RandomForest from the stored fold scores of the submitted file.
    q = np.array(fold_sub)
    rf = np.array(perf["RandomForest"]["cv_5fold"]["mcc_folds"])
    d = q - rf
    t, p = stats.ttest_rel(q, rf)
    hw = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)

    return dict(
        cv_fold_mcc=[round(m, 4) for m in fold_mcc],
        cv_fold_mcc_submitted=fold_sub,
        fold_match=fold_match,
        cv_mcc_mean=round(float(np.mean(fold_mcc)), 4),
        cv_mcc_mean_submitted=sq["cv_5fold"]["mcc_mean"],
        holdout=hold,
        holdout_submitted=hold_sub,
        holdout_max_abs_diff=max(hold_diff.values()),
        holdout_confusion=cm,
        zz_same_range_holdout_mcc=round(zz_hold_mcc, 4),
        classical=classical,
        logreg_full_range=dict(cv_mcc=round(float(np.mean(lr_full)), 4), holdout_mcc=round(lr_full_hold, 4)),
        paired_vs_rf=dict(
            delta_mcc=round(float(d.mean()), 4),
            ci_95=[round(float(d.mean() - hw), 4), round(float(d.mean() + hw), 4)],
            t=round(float(t), 4),
            p=round(float(p), 4),
        ),
    )


# ------------------------------------------------------------------------ sweeps
def _clf(C=1, cw=None):
    return SVC(kernel="precomputed", C=C, class_weight=cw)


def sweep_e3() -> list[dict[str, Any]]:
    """Depth and input-range sweep (true ZZ kernel, 5-fold CV, holdout)."""
    rows = []
    for variant in ("notime", "withtime"):
        sp = load_split(variant)
        for scale in (1, np.pi, 2 * np.pi):
            for reps in (1, 2):
                for cw in (None, "balanced"):
                    S, St = zz_states(sp.Xtr * scale, reps), zz_states(sp.Xte * scale, reps)
                    Kt, Ke = fidelity_kernel(S, S), fidelity_kernel(St, S)
                    pcv = np.zeros_like(sp.ytr)
                    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Kt, sp.ytr):
                        pcv[b] = _clf(cw=cw).fit(Kt[np.ix_(a, a)], sp.ytr[a]).predict(Kt[np.ix_(b, a)])
                    pt = _clf(cw=cw).fit(Kt, sp.ytr).predict(Ke)
                    rows.append(
                        dict(
                            data=variant,
                            range=("0 to 1" if scale == 1 else "0 to pi" if scale == np.pi else "0 to 2pi"),
                            scale=round(float(scale), 4),
                            depth=reps,
                            class_weight=cw or "none",
                            cv_mcc=round(float(matthews_corrcoef(sp.ytr, pcv)), 4),
                            test_mcc=round(float(matthews_corrcoef(sp.yte, pt)), 4),
                            positives_predicted=int(pt.sum()),
                        )
                    )
    return rows


def sweep_e4() -> list[dict[str, Any]]:
    """Preprocessing and range sweep with repeated 5-fold CV."""
    rows = []
    for variant in ("notime", "withtime"):
        sp = load_split(variant)
        for mode in ("minmax", "quantile", "clip"):
            for scale in (0.3, 0.5, 0.75, 1.0, 1.5):
                Xtr, Xte = sp.Xtr.copy(), sp.Xte_raw.copy()
                if mode == "quantile":
                    q = QuantileTransformer(n_quantiles=100, output_distribution="uniform").fit(Xtr)
                    Xtr, Xte = q.transform(Xtr), q.transform(Xte)
                elif mode == "clip":
                    lo, hi = np.percentile(Xtr, 5, 0), np.percentile(Xtr, 95, 0)
                    Xtr, Xte = np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi)
                    Xtr, Xte = (Xtr - lo) / (hi - lo), (Xte - lo) / (hi - lo)
                else:
                    Xte = np.clip(Xte, 0, 1)
                S, St = zz_states(Xtr * scale), zz_states(Xte * scale)
                Kt, Ke = fidelity_kernel(S, S), fidelity_kernel(St, S)
                ms = []
                for a, b in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(Kt, sp.ytr):
                    p = _clf().fit(Kt[np.ix_(a, a)], sp.ytr[a]).predict(Kt[np.ix_(b, a)])
                    ms.append(matthews_corrcoef(sp.ytr[b], p))
                pt = _clf().fit(Kt, sp.ytr).predict(Ke)
                rows.append(
                    dict(
                        data=variant,
                        mode=mode,
                        scale=scale,
                        cv_mcc=round(float(np.mean(ms)), 4),
                        test_mcc=round(float(matthews_corrcoef(sp.yte, pt)), 4),
                    )
                )
    return rows


# ---------------------------------------------------------------------- leakage
def _kernel_cv(K, y, folds):
    ms, aucs = [], []
    for a, b in folds:
        c = SVC(kernel="precomputed").fit(K[np.ix_(a, a)], y[a])
        ms.append(matthews_corrcoef(y[b], c.predict(K[np.ix_(b, a)])))
        aucs.append(roc_auc_score(y[b], c.decision_function(K[np.ix_(b, a)])))
    return ms, aucs


def _model_cv(make, X, y, folds):
    ms, aucs = [], []
    for a, b in folds:
        c = make().fit(X[a], y[a])
        s = c.predict_proba(X[b])[:, 1] if hasattr(c, "predict_proba") else c.decision_function(X[b])
        ms.append(matthews_corrcoef(y[b], c.predict(X[b])))
        aucs.append(roc_auc_score(y[b], s))
    return ms, aucs


def leakage_study() -> dict[str, Any]:
    """How much does follow-up time carry? Single-feature scores and with/without-time swaps."""
    single: list[dict[str, Any]] = []
    for variant in ("notime", "withtime"):
        sp = load_split(variant)
        for i, col in enumerate(sp.columns):
            auc = roc_auc_score(sp.ytr, sp.Xtr[:, i])
            single.append(
                dict(
                    variant=variant,
                    feature=col,
                    auroc_alone=round(float(max(auc, 1 - auc)), 4),
                    direction="higher value, more deaths" if auc >= 0.5 else "lower value, more deaths",
                )
            )
    sw, st = load_split("withtime"), load_split("notime")
    assert (sw.ytr == st.ytr).all(), "both variants must share one patient order"
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(sw.Xtr, sw.ytr))

    def zz_k(sp):
        S = zz_states(sp.Xtr * 0.3)
        return fidelity_kernel(S, S)

    runs = {
        "QSVM (ZZ kernel, inputs x 0.3)": lambda sp: _kernel_cv(zz_k(sp), sp.ytr, folds),
        "SVM-RBF": lambda sp: _model_cv(lambda: SVC(kernel="rbf"), sp.Xtr, sp.ytr, folds),
        "LogisticRegression": lambda sp: _model_cv(lambda: LogisticRegression(max_iter=1000), sp.Xtr, sp.ytr, folds),
        "RandomForest": lambda sp: _model_cv(lambda: RandomForestClassifier(random_state=42), sp.Xtr, sp.ytr, folds),
    }
    paired = []
    for name, fn in runs.items():
        mw, aw = fn(sw)
        mn, an = fn(st)
        paired.append(
            dict(
                model=name,
                mcc_with_time=round(float(np.mean(mw)), 4),
                mcc_without_time=round(float(np.mean(mn)), 4),
                mcc_drop=round(float(np.mean(mw) - np.mean(mn)), 4),
                auroc_with_time=round(float(np.mean(aw)), 4),
                auroc_without_time=round(float(np.mean(an)), 4),
            )
        )
    return dict(
        single_feature=single,
        paired=paired,
        note="Without-time runs put serum_sodium in the place of time; the folds are the same 15 repeated 5-fold splits.",
    )


# -------------------------------------------------------------------------- guard
def guard_study(variant: str = "withtime", n_repeats: int = 3, seed: int = 1) -> dict[str, Any]:
    """Run the QNeuro Forge Honesty Guard on repeated-CV scores of real data."""
    sp = load_split(variant)
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed).split(sp.Xtr, sp.ytr))
    scores: dict[str, dict[str, list[float]]] = {}

    def add(name, ms, aucs):
        scores[name] = dict(mcc=[float(x) for x in ms], auroc=[float(x) for x in aucs])

    for tag, scale, kind in (("qsvm_zz_x1.0", 1.0, "zz"), ("qsvm_zz_x0.3", 0.3, "zz"), ("qsvm_angle_x1.0", 1.0, "angle")):
        if kind == "zz":
            S = zz_states(sp.Xtr * scale)
            K = fidelity_kernel(S, S)
        else:
            K = product_kernel(sp.Xtr, sp.Xtr, scale)
        add(tag, *_kernel_cv(K, sp.ytr, folds))
    classical = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(random_state=42),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
        "svm_rbf": lambda: SVC(kernel="rbf"),
        "xgboost": lambda: _xgb(42),
    }
    for name, mk in classical.items():
        add(name, *_model_cv(mk, sp.Xtr, sp.ytr, folds))

    def fold_scores(name: str, metric: str) -> FoldScores:
        v = np.array(scores[name][metric])
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return FoldScores(
            config_id=name,
            metric=metric,
            scores=v.tolist(),
            mean_score=float(v.mean()),
            std_score=float(v.std(ddof=1)),
            ci_lower=float(v.mean() - h),
            ci_upper=float(v.mean() + h),
        )

    qnames = [n for n in scores if n.startswith("qsvm")]
    cnames = [n for n in scores if not n.startswith("qsvm")]
    verdicts = {}
    for metric in ("mcc", "auroc"):
        v = evaluate_advantage(
            [fold_scores(n, metric) for n in qnames], [fold_scores(n, metric) for n in cnames], metric=metric
        )
        verdicts[metric] = v.model_dump(mode="json")
    table = []
    for n in qnames + cnames:
        m, a = fold_scores(n, "mcc"), fold_scores(n, "auroc")
        table.append(
            dict(
                model=n,
                kind="quantum" if n.startswith("qsvm") else "classical",
                mcc_mean=round(m.mean_score, 4),
                mcc_ci=[round(m.ci_lower, 4), round(m.ci_upper, 4)],
                auroc_mean=round(a.mean_score, 4),
                auroc_ci=[round(a.ci_lower, 4), round(a.ci_upper, 4)],
            )
        )
    return dict(
        variant=variant,
        protocol=f"5-fold stratified x {n_repeats} repeats = {len(folds)} paired splits on the 239 training patients (seed {seed})",
        table=table,
        verdicts=verdicts,
    )


# ------------------------------------------------------------- holdout sensitivity
def holdout_sensitivity(cm: list[list[int]]) -> dict[str, Any]:
    """How far can MCC move if one or two of the 60 predictions had gone the other way?"""
    (tn, fp), (fn, tp) = cm
    base = mcc_from_confusion(cm)
    moves = {
        "TN->FP": (tn - 1, fp + 1, fn, tp),
        "FP->TN": (tn + 1, fp - 1, fn, tp),
        "FN->TP": (tn, fp, fn - 1, tp + 1),
        "TP->FN": (tn, fp, fn + 1, tp - 1),
    }

    def val(t):
        if min(t) < 0:
            return None
        return mcc_from_confusion([[t[0], t[1]], [t[2], t[3]]])

    one = {k: val(v) for k, v in moves.items()}
    one_vals = [v for v in one.values() if v is not None]
    two_vals = []
    for k1, v1 in moves.items():
        for k2, v2 in moves.items():
            t = (tn + (v1[0] - tn) + (v2[0] - tn), fp + (v1[1] - fp) + (v2[1] - fp),
                 fn + (v1[2] - fn) + (v2[2] - fn), tp + (v1[3] - tp) + (v2[3] - tp))
            x = val(t)
            if x is not None:
                two_vals.append(x)
    return dict(
        baseline=round(base, 4),
        one_patient_min=round(min(one_vals), 4),
        one_patient_max=round(max(one_vals), 4),
        two_patients_min=round(min(two_vals), 4),
        two_patients_max=round(max(two_vals), 4),
    )


# ------------------------------------------------------------ platform gap
def platform_gap_study(n_splits: int = 300, seed: int = 0) -> dict[str, Any]:
    """Could the Bloq E1/E2 collapse be luck of the 191/48 split? Local ZZ kernel, 300 random splits."""
    from sklearn.model_selection import StratifiedShuffleSplit

    out: dict[str, Any] = {}
    for variant in ("notime", "withtime"):
        sp = load_split(variant)
        S = zz_states(sp.Xtr)
        K = fidelity_kernel(S, S)
        mccs, no_deaths = [], 0
        for tr, te in StratifiedShuffleSplit(n_splits, test_size=48 / 239, random_state=seed).split(K, sp.ytr):
            p = SVC(kernel="precomputed").fit(K[np.ix_(tr, tr)], sp.ytr[tr]).predict(K[np.ix_(te, tr)])
            mccs.append(float(matthews_corrcoef(sp.ytr[te], p)))
            no_deaths += int(p.sum() == 0)
        m = np.array(mccs)
        out[variant] = dict(
            splits=n_splits,
            mcc_mean=round(float(m.mean()), 4),
            mcc_p05=round(float(np.percentile(m, 5)), 4),
            mcc_p95=round(float(np.percentile(m, 95)), 4),
            share_mcc_at_most_0_05=round(float((m <= 0.05).mean()), 4),
            splits_predicting_no_deaths=no_deaths,
        )
    return out
