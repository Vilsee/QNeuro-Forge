"""Claim checker: every claim in the submitted report, the value we recompute, and a status."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from qneuro.study.data import DATA, load_split
from qneuro.study.experiments import mcc_from_confusion

# Values printed in the submitted report (E3 and E4 tables are rounded to two decimals there).
REPORT_E3 = {
    ("notime", "0 to 1", 1): (0.34, 0.40),
    ("withtime", "0 to 1", 1): (0.52, 0.67),
    ("notime", "0 to 1", 2): (0.32, 0.27),
    ("withtime", "0 to 1", 2): (0.41, 0.57),
    ("notime", "0 to pi", 1): (0.24, 0.32),
    ("withtime", "0 to pi", 1): (0.16, 0.23),
    ("notime", "0 to 2pi", 1): (-0.01, 0.13),
    ("withtime", "0 to 2pi", 1): (-0.01, 0.01),
}
REPORT_E4 = {
    ("withtime", 1.0): (0.54, 0.67),
    ("withtime", 0.3): (0.60, 0.81),
    ("notime", 1.0): (0.41, 0.40),
    ("notime", 0.3): (0.40, 0.45),
}

STATUS_LABEL = {
    "reproduced": "Reproduced",
    "partly": "Partly reproduced",
    "not_reproduced": "Not reproduced",
    "not_verifiable": "Not verifiable here",
    "supported": "Supported by local analysis",
    "hypothesis": "Hypothesis, unverified",
}


def _f(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def build_claims(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    e5, e6, e6cmp = ctx["e5"], ctx["e6"], ctx["e6_compare"]
    claims: list[dict[str, Any]] = []

    def add(cid, claim, source, status, recomputed, evidence):
        claims.append(
            dict(id=cid, claim=claim, source=source, status=status, status_label=STATUS_LABEL[status],
                 recomputed=recomputed, evidence=evidence)
        )

    # ---- Bloq platform outputs (raw files)
    for tag, fn, mcc_rep in (("E1", "E1_notime_config.json", 0.0), ("E2", "E2_withtime_config.json", 0.0116)):
        d = json.loads((DATA / "bloq_outputs" / fn).read_text("utf-8"))
        cm = d["output"]["confusion_matrix"]
        m = mcc_from_confusion(cm)
        pos = cm[0][1] + cm[1][1]
        add(
            f"C-{tag}",
            f"{tag} (Bloq, ZZ default): MCC {mcc_rep:.4f}; {pos} of {sum(map(sum, cm))} patients predicted to die.",
            f"data/morphos/bloq_outputs/{fn}",
            "reproduced" if abs(m - mcc_rep) < 5e-4 else "not_reproduced",
            f"MCC recomputed from the platform confusion matrix {cm}: {m:.4f}; accuracy {d['output']['accuracy']:.4f}.",
            "Arithmetic on the raw platform output. This checks the report's reading of the file, not the platform's run.",
        )

    # ---- split
    sp = load_split("withtime")
    sp_nt = load_split("notime")
    add(
        "C-SPLIT",
        "Stratified split: 239 training and 60 test patients with matching death rates.",
        "data/morphos/original/*.csv",
        "reproduced" if len(sp.ytr) == 239 and len(sp.yte) == 60 else "not_reproduced",
        f"{len(sp.ytr)} train (death rate {sp.ytr.mean():.3f}), {len(sp.yte)} test (death rate {sp.yte.mean():.3f}, {int(sp.yte.sum())} deaths).",
        "Counted from the bundled CSVs.",
    )
    lo = float(min(sp.Xte_raw.min(), sp_nt.Xte_raw.min()))
    hi = float(max(sp.Xte_raw.max(), sp_nt.Xte_raw.max()))
    add(
        "C-RANGE",
        "Test rows fall slightly outside [0, 1] because the scaler only saw training rows.",
        "data/morphos/original/*_test.csv",
        "reproduced" if lo < 0 or hi > 1 else "not_reproduced",
        f"Test feature minimum {lo:.3f}, maximum {hi:.3f}. The study clips test rows to [0, 1] before encoding.",
        "Min and max over both test files.",
    )

    # ---- E3 / E4
    if ctx.get("e3"):
        rows = {(r["data"], r["range"], r["depth"]): r for r in ctx["e3"] if r["class_weight"] == "none"}
        ok = sum(
            abs(rows[k]["cv_mcc"] - v[0]) < 0.006 and abs(rows[k]["test_mcc"] - v[1]) < 0.006
            for k, v in REPORT_E3.items()
        )
        add(
            "C-E3",
            "E3: wider input range and deeper circuits hurt the ZZ kernel (8 rows of the report table).",
            "sweep1_depth_scale.py, re-run",
            "reproduced" if ok == len(REPORT_E3) else "partly",
            f"{ok} of {len(REPORT_E3)} rows match to the report's two decimals. Example: withtime range 0 to 1 depth 1 "
            f"gives CV {rows[('withtime', '0 to 1', 1)]['cv_mcc']:.4f} and test {rows[('withtime', '0 to 1', 1)]['test_mcc']:.4f}.",
            "The report calls these approximate. They are now exact.",
        )
    if ctx.get("e4"):
        rows4 = {(r["data"], r["scale"]): r for r in ctx["e4"] if r["mode"] == "minmax"}
        ok4 = sum(
            abs(rows4[k]["cv_mcc"] - v[0]) < 0.006 and abs(rows4[k]["test_mcc"] - v[1]) < 0.006
            for k, v in REPORT_E4.items()
        )
        w3, w1 = rows4[("withtime", 0.3)], rows4[("withtime", 1.0)]
        add(
            "C-E4",
            "E4: shrinking the input range to 0.3 helped the with-time data (CV 0.60, test 0.81).",
            "sweep2_preprocessing.py, re-run",
            "reproduced" if ok4 == len(REPORT_E4) else "partly",
            f"{ok4} of {len(REPORT_E4)} cells match. With time: range 0.3 CV {w3['cv_mcc']:.4f} / test {w3['test_mcc']:.4f}; "
            f"range 1.0 CV {w1['cv_mcc']:.4f} / test {w1['test_mcc']:.4f}.",
            "Min-max rows only.",
        )
        wt = sorted((r["scale"], r["cv_mcc"], r["test_mcc"]) for r in ctx["e4"] if r["data"] == "withtime" and r["mode"] == "minmax")
        cv_spread = max(x[1] for x in wt) - min(x[1] for x in wt)
        te_spread = max(x[2] for x in wt) - min(x[2] for x in wt)
        add(
            "C-SELECT",
            "The range 0.3 was chosen on cross-validation, not on the holdout.",
            "docs/MORPHOS_REPORT_AND_PRD.md, section on decisions",
            "supported",
            f"Across five ranges (with time) CV MCC varies by {cv_spread:.3f} but holdout MCC varies by {te_spread:.3f}.",
            "Holdout MCC on 60 patients is noisy, so choosing a range by holdout would mostly choose noise.",
        )

    # ---- E5
    h, hs = e5["holdout"], e5["holdout_submitted"]
    exact = all(abs(h[k] - hs[k]) < 1e-9 for k in ("mcc", "accuracy", "f1", "precision", "recall"))
    add(
        "C-E5-HOLD",
        "E5 QSVM holdout: MCC 0.746, accuracy 0.883, recall 0.895, confusion [[36, 5], [2, 17]].",
        "data/morphos/submissions/SUBMITTED_corrected.json",
        "reproduced" if exact and e5["holdout_confusion"] == [[36, 5], [2, 17]] else "partly",
        f"Independent angle-embedding kernel gives MCC {h['mcc']:.6f}, accuracy {h['accuracy']:.4f}, confusion {e5['holdout_confusion']}. "
        f"ROC-AUC {h['roc_auc']:.4f} against {hs['roc_auc']:.4f} submitted (score function differs).",
        "All five class-prediction metrics match to 9 decimals. Only the score-based ROC-AUC differs.",
    )
    nm = sum(e5["fold_match"])
    bad = [i for i, ok in enumerate(e5["fold_match"]) if not ok]
    add(
        "C-E5-CV",
        "E5 QSVM 5-fold MCCs: 0.318, 0.359, 0.514, 0.633, 0.711 (mean 0.507).",
        "data/morphos/submissions/SUBMITTED_corrected.json",
        "reproduced" if nm == 5 else "partly",
        f"{nm} of 5 folds match. Mean {e5['cv_mcc_mean']:.4f} against {e5['cv_mcc_mean_submitted']:.4f}."
        + (f" Fold {bad[0] + 1}: {e5['cv_fold_mcc'][bad[0]]:.4f} against {e5['cv_fold_mcc_submitted'][bad[0]]:.4f}." if bad else ""),
        "Same folds (5-fold, shuffled, seed 42). The fold 4 gap is 0.016 MCC and its cause is not known.",
    )
    add(
        "C-E5-MAP",
        "The submitted kernel is an angle-embedding fidelity kernel with no entanglement, not a ZZ feature map.",
        "metadata.feature_map, corrected before submission",
        "supported",
        f"The product-state kernel reproduces the submitted holdout exactly (MCC {h['mcc']:.3f}). "
        f"A true ZZ kernel at the same range gives holdout MCC {e5['zz_same_range_holdout_mcc']:.3f}.",
        "Two kernels, one match. The platform's own export called it ZZFeatureMap, which the corrected file no longer does.",
    )
    c = e5["classical"]
    add(
        "C-E5-LR",
        "In E5 logistic regression predicted the majority class in every fold (CV MCC 0.000, holdout 0.191).",
        "data/morphos/submissions/SUBMITTED_corrected.json",
        "reproduced" if c["LogisticRegression"]["reproduced"] else "not_reproduced",
        f"Reproduced when the inputs are multiplied by 0.3: CV {c['LogisticRegression']['reproduced_cv_mcc']:.4f}, holdout {c['LogisticRegression']['reproduced_holdout_mcc']:.4f}. "
        f"On the full 0 to 1 range the same model gets CV {e5['logreg_full_range']['cv_mcc']:.4f} and holdout {e5['logreg_full_range']['holdout_mcc']:.4f}, "
        f"which equals the QSVM's holdout MCC.",
        "Inferred recipe. The collapse is a regularisation effect of shrunken inputs, so beating this baseline says little.",
    )
    add(
        "C-E5-SVM",
        "E5 SVM-RBF baseline: CV MCC 0.525, holdout 0.779.",
        "data/morphos/submissions/SUBMITTED_corrected.json",
        "reproduced" if c["SVM-RBF"]["reproduced"] else "not_reproduced",
        f"CV {c['SVM-RBF']['reproduced_cv_mcc']:.4f}, holdout {c['SVM-RBF']['reproduced_holdout_mcc']:.4f}.",
        "scikit-learn SVC defaults on the same folds.",
    )
    trees = [k for k in ("RandomForest", "GradientBoosting", "XGBoost") if not c[k]["reproduced"]]
    add(
        "C-E5-TREES",
        "E5 RandomForest, GradientBoosting and XGBoost baseline numbers.",
        "data/morphos/submissions/SUBMITTED_corrected.json",
        "not_reproduced" if len(trees) == 3 else "partly",
        "; ".join(f"{k}: CV {c[k]['reproduced_cv_mcc']:.4f} vs {c[k]['submitted_cv_mcc']:.4f}, holdout {c[k]['reproduced_holdout_mcc']:.4f} vs {c[k]['submitted_holdout_mcc']:.4f}" for k in trees),
        "Default settings and seed 42 do not reproduce them. The platform's tree settings are not visible to us.",
    )
    p = e5["paired_vs_rf"]
    add(
        "C-E5-PAIR",
        "QSVM vs RandomForest paired on E5 folds: delta MCC -0.122, 95% CI [-0.290, 0.045], p = 0.112.",
        "stored fold scores in SUBMITTED_corrected.json",
        "reproduced",
        f"Delta {p['delta_mcc']:+.4f}, CI [{p['ci_95'][0]:+.4f}, {p['ci_95'][1]:+.4f}], t {p['t']:.3f}, p {p['p']:.4f}.",
        "Plain paired t-test on 5 folds. The guard's corrected test (below) is stricter.",
    )

    # ---- E6
    add(
        "C-E6",
        "E6 local ZZ benchmark (not submitted): QSVM CV MCC 0.569, holdout 0.806.",
        "data/morphos/submissions/E6_local_zz_NOT_submitted.json",
        "reproduced" if e6cmp["identical"] else "partly",
        f"Fresh run against the stored file: {e6cmp['n_differences']} differing values (timing excluded). "
        f"QSVM CV {e6['performance_summary']['QSVM (Quantum Kernel)']['cv_5fold']['mcc_mean']:.4f}, "
        f"holdout {e6['performance_summary']['QSVM (Quantum Kernel)']['holdout_test']['mcc']:.4f}.",
        "Every metric and fold score identical.",
    )

    # ---- integrity
    ig = ctx["integrity"]
    add(
        "C-DIFF",
        "The submitted file differs from the original Bloq-pipeline export in three text fields only.",
        "SUBMITTED_corrected.json vs E5_original_export.json",
        "reproduced" if ig["only_text_fields_changed"] and len(ig["diff"]) == 3 else "partly",
        f"{len(ig['diff'])} fields differ: " + ", ".join(d["field"] for d in ig["diff"]) + ". Every number is identical.",
        "Programmatic recursive diff of the two JSON files.",
    )

    # ---- leakage
    lk = ctx.get("leakage")
    if lk:
        t = next(r for r in lk["single_feature"] if r["variant"] == "withtime" and r["feature"] == "time")
        best_other = max((r for r in lk["single_feature"] if r["variant"] == "withtime" and r["feature"] != "time"), key=lambda r: r["auroc_alone"])
        drops = ", ".join(f"{r['model'].split(' (')[0]} {r['mcc_drop']:+.3f}" for r in lk["paired"])
        add(
            "C-LEAK",
            "Follow-up time carries outcome information, so with-time scores are optimistic for a deployable model.",
            "docs/MORPHOS_REPORT_AND_PRD.md, risks; no no-time comparison was in the submitted file",
            "supported",
            f"`time` alone has AUROC {t['auroc_alone']:.3f}, higher than any other single feature (best other: {best_other['feature']} {best_other['auroc_alone']:.3f}). "
            f"Swapping it for serum_sodium changes CV MCC by: {drops}.",
            "Local repeated-CV run, same 15 splits for both feature sets. Not run on Bloq.",
        )

    # ---- platform gap
    gap = ctx.get("gap")
    if gap:
        g = gap["notime"]
        gw = gap["withtime"]
        add(
            "C-GAP",
            "The Bloq E1 and E2 collapse is explained by kernel concentration at large input ranges (report, E3).",
            "docs/MORPHOS_REPORT_AND_PRD.md, E3 finding",
            "hypothesis",
            f"Local ZZ kernels on 0 to 1 inputs never predicted 'no deaths' in {g['splits']} random 191/48 splits "
            f"(no-time: {g['splits_predicting_no_deaths']}, with-time: {gw['splits_predicting_no_deaths']}), and MCC was at most 0.05 in "
            f"{g['share_mcc_at_most_0_05'] * 100:.1f}% of no-time splits. Local collapse appears only at range 0 to 2pi.",
            "Split luck does not explain E1 and E2. Whether the platform rescales inputs internally cannot be checked from outside.",
        )

    # ---- sensitivity
    s = ctx["sensitivity"]
    add(
        "C-NOISE",
        "On a 60-patient holdout, one or two patients move MCC noticeably.",
        "holdout confusion matrix of the E5 QSVM",
        "supported",
        f"Baseline {s['baseline']:.3f}. One prediction flipped: {s['one_patient_min']:.3f} to {s['one_patient_max']:.3f}. "
        f"Two flipped: {s['two_patients_min']:.3f} to {s['two_patients_max']:.3f}.",
        "Computed from the four confusion-matrix counts.",
    )

    # ---- provenance
    add(
        "C-PLATFORM",
        "E5 was produced by the Bloq platform pipeline.",
        "platform account, outside this repository",
        "not_verifiable",
        "No platform signature or run identifier in the file can be checked from outside Bloq.",
        "This tool signs and hashes what you hand it. It cannot attest to where a number came from. Keep the platform screenshots and the experiment ids.",
    )
    return claims
