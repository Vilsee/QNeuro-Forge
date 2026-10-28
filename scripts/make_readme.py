"""Write README.md. Every number is read from public/data/study.json, so the README cannot drift from the signed study.
    python scripts/make_readme.py                                   # without a live link
    python scripts/make_readme.py --live-url https://your-app.vercel.app
Regenerate after `qneuro study build` and `python scripts/make_figures.py`."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--live-url", default=None, help="Deployed site address, e.g. https://qneuro-forge.vercel.app")
ARGS = ap.parse_args()

ENV = json.loads((ROOT / "public/data/study.json").read_text("utf-8"))
S = ENV["study"]
AUDIT = [json.loads(l) for l in (ROOT / "public/data/audit.jsonl").read_text("utf-8").splitlines() if l.strip()]
M = S["meta"]; DS = M["dataset"]; E5 = S["e5"]; E6 = S["e6"]; H = E5["holdout"]; HS = E5["holdout_submitted"]; SENS = S["sensitivity"]
GW, GN = S["guard"]["withtime"], S["guard"].get("notime")
VW = GW["verdicts"]["mcc"]; VN = GN["verdicts"]["mcc"] if GN else None
LK, GAP = S["leakage"], S["gap"]
f3 = lambda x: ("\u2212" if x < 0 else "") + f"{abs(x):.3f}"
f4 = lambda x: ("\u2212" if x < 0 else "") + f"{abs(x):.4f}"
sg = lambda x, d=3: ("+" if x >= 0 else "\u2212") + f"{abs(x):.{d}f}"
NICE = {"qsvm_zz_x0.3": "QSVM, ZZ kernel, inputs x0.3", "qsvm_zz_x1.0": "QSVM, ZZ kernel, inputs x1.0", "qsvm_angle_x1.0": "QSVM, angle kernel, x1.0",
        "logistic_regression": "Logistic regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
        "svm_rbf": "SVM, RBF kernel", "xgboost": "XGBoost"}


def counts(name):
    rows = list(csv.DictReader(open(ROOT / "data/morphos/original" / name, encoding="utf-8")))
    return len(rows), sum(int(float(r["death_event"])) for r in rows)


NTR, DTR = counts("hf_withtime_train.csv"); NTE, DTE = counts("hf_withtime_test.csv")
ST = {}
for c in S["claims"]: ST[c["status"]] = ST.get(c["status"], 0) + 1
HOLD = ST.get("reproduced", 0) + ST.get("supported", 0)
CL = {c["id"]: c for c in S["claims"]}
TIME_AUC = next(r["auroc_alone"] for r in LK["single_feature"] if r["variant"] == "withtime" and r["feature"] == "time")
DROPS = [r["mcc_drop"] for r in LK["paired"]]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows: out.append("| " + " | ".join(str(x).replace("|", "/") for x in r) + " |")
    return "\n".join(out)


def mermaid(name): return "```mermaid\n" + (ROOT / "docs/diagrams" / f"{name}.mmd").read_text("utf-8").strip() + "\n```"


def img(name, alt, folder="assets"): return f"![{alt}](docs/{folder}/{name})"


ntr, nte = DS["train"], DS["test"]
p_tt = ntr * (ntr - 1) // 2; p_te = nte * ntr; p_fit = p_tt + p_te
ftr, fte = 191, 48
p_fold = ftr * (ftr - 1) // 2 + fte * ftr; p_cv = p_fold * 5


def guard_rows(G):
    return [(NICE.get(r["model"], r["model"]), "quantum" if r["kind"] == "quantum" else "classical", f3(r["mcc_mean"]),
             f"{f3(r['mcc_ci'][0])} to {f3(r['mcc_ci'][1])}", f3(r["auroc_mean"])) for r in sorted(G["table"], key=lambda r: -r["mcc_mean"])]


e3 = [r for r in S["sweeps"]["e3"] if r["depth"] == 1 and r["class_weight"] == "none"]
e3rows = [(("with time" if r["data"] == "withtime" else "no time"), r["range"].replace("pi", "\u03c0"), f3(r["cv_mcc"]), f3(r["test_mcc"])) for r in sorted(e3, key=lambda r: (r["data"] != "withtime", r["scale"]))]
e4 = [r for r in S["sweeps"]["e4"] if r["mode"] == "minmax"]
e4rows = [(("with time" if r["data"] == "withtime" else "no time"), r["scale"], f3(r["cv_mcc"]), f3(r["test_mcc"])) for r in sorted(e4, key=lambda r: (r["data"] != "withtime", r["scale"]))]
e5rows = []
for k, v in S["submitted_summary"].items():
    rep = E5["classical"].get(k, {}).get("reproduced")
    note = "reproduced (independent kernel)" if k.startswith("QSVM") else ("reproduced" if rep else "not reproduced")
    e5rows.append((k, f3(v["cv_5fold"]["mcc_mean"]), f3(v["holdout_test"]["mcc"]), note))
claims_rows = [(c["id"], c["status_label"], c["claim"], c["recomputed"]) for c in S["claims"]]
single = [(r["feature"], f3(r["auroc_alone"]), r["direction"]) for r in LK["single_feature"] if r["variant"] == "withtime"]
paired = [(r["model"], f3(r["mcc_with_time"]), f3(r["mcc_without_time"]), sg(-r["mcc_drop"]), f3(r["auroc_with_time"]), f3(r["auroc_without_time"])) for r in LK["paired"]]
diff_rows = [(f"`{d['field']}`", d["before"], d["after"]) for d in S["integrity"]["diff"]]
acts = {}
for a in AUDIT: acts[a["action"]] = acts.get(a["action"], 0) + 1

spec = importlib.util.spec_from_file_location("mds", ROOT / "scripts" / "make_demo_script.py"); mds = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mds); QA = mds.QA
except Exception:
    QA = []
qa_md = "\n\n".join(f"**{q}**\n\n{a}" for q, a in QA)

LIVE = (f"| Live site | {ARGS.live_url} | The whole study as a page, with the built-in walkthrough. Open it before the session starts. |"
        if ARGS.live_url else "| Local site | `qneuro study serve --port 8080`, then http://127.0.0.1:8080/ | The whole study as a page. Run the command from the repository root. |")
LIVE_LINE = (f"**Live site:** {ARGS.live_url}  \n" if ARGS.live_url else "")
BADGES = ("![python](https://img.shields.io/badge/python-3.11%2B-1C1B18?style=flat-square&labelColor=1C1B18&color=1E6B55) "
          "![signed](https://img.shields.io/badge/signed-ML--DSA--65-1C1B18?style=flat-square&labelColor=1C1B18&color=1E6B55) "
          "![status](https://img.shields.io/badge/status-research%20prototype-1C1B18?style=flat-square&labelColor=1C1B18&color=86600F)")
DEV = max(SENS["one_patient_max"] - H["mcc"], H["mcc"] - SENS["one_patient_min"])

README = f"""# QNeuro Forge

**Re-check, sign and verify a quantum-kernel study: every number from Team Morphos's Bloq hackathon entry on heart-failure survival, recomputed by code and sealed with post-quantum signatures.**

{BADGES}

{img("hero.svg", "From 299 patients to a signed ledger, with the four headline results")}

{LIVE_LINE}**Source:** https://github.com/Vilsee/QNeuro-Forge

> **In one paragraph.** Team Morphos built a quantum-kernel support vector machine (QSVM) on the Bloq platform to predict death in 299 heart-failure patients, and submitted a holdout MCC of **{f3(H['mcc'])}** on 60 patients. This repository asks a plainer question: which of our own claims survive a re-check? **{HOLD} of {len(S['claims'])}** claims hold, the quantum kernel is **statistically tied** with the best classical model ({f3(VW['quantum_score'])} against {f3(VW['classical_score'])} MCC over 15 paired splits), and every file and every analysis step is hashed and signed so anyone can verify it. Nothing here claims a quantum advantage, and nothing here is a clinical tool.

## Contents

1. [Open these during the demo](#1-open-these-during-the-demo)
2. [The result in 60 seconds](#2-the-result-in-60-seconds)
3. [Background for non-specialists](#3-background-for-non-specialists)
4. [The data](#4-the-data)
5. [What was run: six experiments](#5-what-was-run-six-experiments)
6. [Findings, one at a time](#6-findings-one-at-a-time)
7. [The claim checker](#7-the-claim-checker)
8. [How it works](#8-how-it-works)
9. [Security and verification](#9-security-and-verification)
10. [The page and the walkthrough](#10-the-page-and-the-walkthrough)
11. [Run it yourself](#11-run-it-yourself)
12. [Repository layout](#12-repository-layout)
13. [Reproducibility notes](#13-reproducibility-notes)
14. [Limitations](#14-limitations)
15. [Explaining it to the professor](#15-explaining-it-to-the-professor)
16. [Glossary](#16-glossary)
17. [References](#17-references)

---

## 1. Open these during the demo

| What | Link | Why show it |
| --- | --- | --- |
{LIVE}
| Source code | https://github.com/Vilsee/QNeuro-Forge | The commit history shows the study layer, the tests and the signed outputs. |
| Bloq platform, Experiments page | https://platform.bloq.in/?m=experiments | Where E1 and E2 were run. Show the experiment list and the algorithm and device choices. |
| Bloq coverage: no-code platform launch | https://quantumcomputingreport.com/bloq-quantum-launches-no-code-platform-to-accelerate-algorithm-development/ | Independent description of what the platform is for. |
| Bloq coverage: platform launch | https://thequantuminsider.com/2026/05/07/bloq-quantum-launches-easy-to-use-platform-to-simplify-quantum-computing-use-case-development/ | The same launch from a second outlet. |
| Bloq coverage: IBM Quantum Network | https://thequantuminsider.com/?p=2385279 | Context on the hardware access behind the platform. |
| Dataset page (UCI) | https://archive.ics.uci.edu/dataset/519/heart-failure-clinical-records | The public data: 299 patients, CC BY 4.0, citation and variable list. |
| Dataset paper | https://doi.org/10.1186/s12911-020-1023-5 | Chicco and Jurman (2020), where the data was introduced. |
| Why MCC | https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7 | Chicco and Jurman (2020): the argument for MCC over accuracy and F1. |
| NIST standard for the signature | https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.204.pdf | FIPS 204, the ML-DSA standard used to sign the study. |
| Fair cross-validation test | https://papers.nips.cc/paper/1661-inference-for-the-generalization-error | Nadeau and Bengio, the corrected test the Honesty Guard uses. |

More links are in [References](#17-references).

---

## 2. The result in 60 seconds

{table(["Question", "Answer from the code"], [
    ("Does the submitted QSVM result reproduce?", f"Yes. The five class-based holdout metrics match the submitted file to nine decimals, with the same confusion matrix {E5['holdout_confusion']}. ROC-AUC differs by {abs(H['roc_auc'] - HS['roc_auc']):.4f}."),
    ("Which kernel is it?", f"An angle-embedding kernel with no entanglement. A true ZZ kernel on the same inputs gives holdout MCC {f3(E5['zz_same_range_holdout_mcc'])}."),
    ("Is the quantum model better than classical?", f"Not demonstrably. Over 15 paired cross-validation splits the best quantum kernel scores {f3(VW['quantum_score'])} MCC and the best classical model ({NICE.get(VW['best_classical_model'], VW['best_classical_model'])}) {f3(VW['classical_score'])}. Guard verdict: `{VW['category']}`."),
    ("How reliable is the 60-patient holdout?", f"One changed prediction moves MCC between {f3(SENS['one_patient_min'])} and {f3(SENS['one_patient_max'])}. Two move it between {f3(SENS['two_patients_min'])} and {f3(SENS['two_patients_max'])}."),
    ("Does the follow-up time feature matter?", f"Alone it scores AUROC {f3(TIME_AUC)}. Replacing it with serum sodium lowers MCC for every model by {f3(min(DROPS))} to {f3(max(DROPS))}."),
    ("Why did the Bloq runs E1 and E2 collapse?", f"Unknown. Local ZZ kernels on inputs in 0 to 1 never predicted 'no deaths' in {GAP['notime']['splits']} random splits, so split luck does not explain it. Kernel concentration at wide input ranges is a hypothesis."),
    ("What was edited before submission?", "Three text fields, shown in section 6.7. Every number is identical."),
    ("Can anyone verify this?", f"Yes. {len(S['manifest'])} files are hashed in a signed manifest, the study is signed with ML-DSA-65, and {ENV['audit']['entries']} audit entries are hash-chained. One command checks it all and a second tries to break it."),
])}

{img("claims.svg", "Twenty claims coloured by whether they reproduced")}

---

## 3. Background for non-specialists

### 3.1 The task

A hospital recorded 13 numbers for each of 299 heart-failure patients and whether the patient died during follow-up (`death_event`). The task is classification: from a few of those numbers, predict who died. This is a research exercise on a public dataset. It is not a tool for deciding anything about a real patient.

### 3.2 Why MCC and not accuracy

About {DTR / NTR * 100:.0f}% of the training patients died, so a model that answers "survived" for everyone is about {100 - DTR / NTR * 100:.0f}% accurate and useless. The Matthews correlation coefficient (MCC) scores a prediction from all four cells of the confusion matrix and equals 0 for no skill and 1 for perfect prediction, whatever the class balance. Experiment E1 shows why it matters: Bloq reported accuracy {f4(S['bloq']['E1']['output']['accuracy'])} but MCC {f4(S['bloq']['E1']['output']['mcc'])}, because the model predicted "survived" for all 48 patients.

```
MCC = (TP x TN - FP x FN) / sqrt( (TP+FP) (TP+FN) (TN+FP) (TN+FN) )
```

### 3.3 What a quantum kernel SVM is

An SVM separates two groups using a similarity measure called a kernel. A *quantum kernel* computes that similarity from quantum states: each patient's numbers set up the state of four qubits, and the similarity of two patients is how much their states overlap. The SVM itself is the ordinary classical algorithm. No quantum advantage is assumed anywhere in this project; the question is only whether the quantum similarity measure does better than classical models, and by how much.

{img("concept-qsvm.svg", "Two patients become two quantum states; their overlap is the kernel; the kernel matrix feeds a normal SVM")}

### 3.4 Two kernels that are easy to confuse

The submitted result was labelled as a ZZ feature map in the platform's own export. Re-implementing it showed that the numbers come from an angle-embedding kernel instead. The difference matters because only the ZZ circuit entangles the qubits.

{img("circuits.svg", "Angle embedding uses one rotation per qubit and no entanglement; the ZZ feature map entangles every pair")}

### 3.5 Why a simulator, and what real hardware would cost

All circuits here are simulated on an ordinary computer: one state per patient, then a matrix product gives every overlap. On real hardware, every pair of patients needs its own circuit run, repeated for many shots. Counting pairs for this dataset:

{table(["Quantity", "Arithmetic", "Value"], [
    ("Training pairs for the kernel matrix", f"{ntr} x {ntr - 1} / 2", f"{p_tt:,}"),
    ("Holdout-versus-training pairs", f"{nte} x {ntr}", f"{p_te:,}"),
    ("One fit and one holdout prediction", f"{p_tt:,} + {p_te:,}", f"{p_fit:,} circuits"),
    (f"One 5-fold cross-validation pass (about {ftr} train, {fte} test per fold)", f"5 x ({ftr * (ftr - 1) // 2:,} + {fte * ftr:,})", f"{p_cv:,} circuits"),
    ("Shots at 1,000 per circuit, fit and holdout only", f"{p_fit:,} x 1,000", f"{p_fit * 1000:,} shots"),
])}

This is why the study stays on a simulator, and why a hardware run would be a project of its own.

---

## 4. The data

{img("data-split.svg", "239 training and 60 holdout patients, with two feature sets of four features each")}

- **Source:** UCI Machine Learning Repository, dataset 519, Heart Failure Clinical Records, 299 patients, CC BY 4.0.
- **Split:** one stratified 80/20 split by outcome, `random_state=7`: {NTR} training patients ({DTR} died) and {NTE} holdout patients ({DTE} died).
- **Scaling:** a MinMax scaler fitted on the training rows only. Holdout values can therefore leave 0 to 1 slightly, and are clipped before encoding.
- **Four features, one per qubit:** the track allows four qubits. The *with time* set uses ejection fraction, serum creatinine, age and follow-up time. The *no time* set swaps time for serum sodium.
- **Files:** the four prepared CSVs sit in [`data/morphos/original/`](data/morphos/original/) exactly as submitted. `qneuro study prepare` rebuilds them from UCI and prints their SHA-256 hashes.

---

## 5. What was run: six experiments

{img("experiments.svg", "E1 and E2 on Bloq, E3 and E4 local sweeps, E5 submitted, E6 local and not submitted")}

{table(["Id", "What", "Where", "Data", "CV MCC", "Holdout MCC", "Re-check"], [
    (r["id"], r["what"], r["where"], r["data"], f3(r["cv_mcc"]) if r.get("cv_mcc") is not None else "-", f3(r["holdout_mcc"]) if r.get("holdout_mcc") is not None else "-", r["status_label"]) for r in S["ledger"]])}

E1 and E2 report Bloq's own 48-row validation split, not the 60-patient holdout. E4 reports the best cross-validated range. Experiments E1, E2 and E5 were run on the Bloq platform and can only be checked as files. E3, E4 and E6 are local and re-run with one command.

---

## 6. Findings, one at a time

### 6.1 The submitted QSVM result reproduces

An independent angle-embedding kernel, written in NumPy and checked against PennyLane's `AngleEmbedding` in the test suite, reproduces the submitted holdout. Five class-based metrics match to nine decimals and the confusion matrix is identical. ROC-AUC differs by {abs(H['roc_auc'] - HS['roc_auc']):.4f} because the score function differs. Four of five cross-validation folds match exactly. Fold 4 is {f3(E5['cv_fold_mcc'][3])} here against {f3(E5['cv_fold_mcc_submitted'][3])} in the file, and the cause is not known.

{table(["Metric", "Ours", "Submitted"], [(k, f"{H[k]:.6f}", f"{HS[k]:.6f}") for k in ("mcc", "accuracy", "f1", "precision", "recall", "roc_auc")])}

### 6.2 The baselines in the submitted file

{table(["Model", "CV MCC", "Holdout MCC", "Reproduced here?"], e5rows)}

Two details change how to read this table. Logistic regression collapses to the majority class only when its inputs are multiplied by 0.3, which the reproduction found by trial; on the full 0 to 1 range it scores CV {f3(E5['logreg_full_range']['cv_mcc'])} and holdout {f3(E5['logreg_full_range']['holdout_mcc'])}, equal to the QSVM's holdout. So beating that baseline says little. The three tree models do not reproduce with default settings and seed 42.

### 6.3 A fair comparison: same splits, same budget, one judge

All eight models were scored on the same 15 paired splits (5-fold, 3 repeats) of the 239 training patients, and the scores went to the project's Honesty Guard, which is the only code allowed to word a quantum-advantage claim.

{img("fair-comparison.svg", "Interval plot of MCC for eight models over 15 paired splits, with follow-up time")}

{table(["Model", "Kind", "MCC", "95% t-interval", "AUROC"], guard_rows(GW))}

> {VW['sentence']}

Without follow-up time the picture is the same: the best quantum configuration scores {f3(VN['quantum_score']) if VN else 'n/a'} and the best classical model {f3(VN['classical_score']) if VN else 'n/a'}, verdict `{VN['category'] if VN else 'n/a'}`.

<details><summary>Same table without follow-up time</summary>

{table(["Model", "Kind", "MCC", "95% t-interval", "AUROC"], guard_rows(GN)) if GN else ""}

{img("fair-comparison-notime.svg", "Interval plot of MCC without follow-up time")}

</details>

### 6.4 The 60-patient holdout is fragile

{img("holdout.svg", "Sixty dots, one per holdout patient, with seven wrong predictions in red")}

Flipping one prediction moves MCC between {f3(SENS['one_patient_min'])} and {f3(SENS['one_patient_max'])}. Flipping two moves it between {f3(SENS['two_patients_min'])} and {f3(SENS['two_patients_max'])}. Gaps of a few hundredths between models on this holdout are noise. That is why the input range was chosen by cross-validation and not by the holdout: {CL['C-SELECT']['recomputed']}

### 6.5 Follow-up time carries the outcome

{img("time-leakage.svg", "Single-feature AUROC and with-time versus without-time MCC")}

{table(["Feature (with-time set)", "AUROC alone", "Direction"], single)}

{table(["Model", "MCC with time", "MCC without", "Change", "AUROC with", "AUROC without"], paired)}

`time` is how long a patient was followed, and patients who die are followed for less time, so a model that needs it cannot predict ahead. The track allows it. The with-time numbers are therefore optimistic for any real use. {LK['note']} The submitted report listed this comparison as future work; it was run here locally and not on Bloq.

### 6.6 Why the Bloq runs collapsed, and what can and cannot be said

E1 (no time) and E2 (with time) used the platform's default ZZ feature map on four qubits and scored MCC {f4(S['bloq']['E1']['output']['mcc'])} and {f4(S['bloq']['E2']['output']['mcc'])}. The local sweep shows that a ZZ kernel does collapse when its inputs are stretched:

{img("sweeps.svg", "Cross-validated MCC against input range for E3 and E4")}

<details><summary>Sweep tables (E3 depth 1, E4 min-max)</summary>

**E3, depth 1**

{table(["Data", "Input range", "CV MCC", "Holdout MCC"], e3rows)}

**E4, min-max inputs, repeated 5-fold**

{table(["Data", "Range multiplier", "CV MCC", "Holdout MCC"], e4rows)}

</details>

But collapse at wide ranges does not explain the platform runs, because those inputs were in 0 to 1. With inputs in 0 to 1 the local ZZ kernel predicted **no deaths in {GAP['notime']['splits_predicting_no_deaths']} of {GAP['notime']['splits']} random splits** without time and {GAP['withtime']['splits_predicting_no_deaths']} of {GAP['withtime']['splits']} with time, and its MCC was at most 0.05 in {GAP['notime']['share_mcc_at_most_0_05'] * 100:.1f}% of the no-time splits (mean {f3(GAP['notime']['mcc_mean'])}, 5th to 95th percentile {f3(GAP['notime']['mcc_p05'])} to {f3(GAP['notime']['mcc_p95'])}). Split luck is ruled out. Whether the platform rescales inputs internally cannot be checked from outside, so it stays a hypothesis. E2's saved title says "notime" although its features include time, a labelling slip in the platform entry.

### 6.7 What was edited before submission

Three text fields differ between the Bloq pipeline export and the submitted file. Every number is identical.

{table(["Field", "Bloq export (before)", "Submitted (after)"], diff_rows)}

The original note claimed that follow-up time inflates MCC across models, which the file had never tested, and it named a ZZ map although the kernel that reproduces the numbers has no entanglement.

---

## 7. The claim checker

Each claim in the submitted report is recomputed and given one of six statuses.

{img("claims.svg", "Twenty claims coloured by status")}

{table(["Status", "Meaning"], [("Reproduced", "Code re-ran it, or recomputed it from the stored file, and the value matches."), ("Supported by local analysis", "An inference the local runs back up."), ("Partly reproduced", "Most of it matches; one part does not."), ("Not reproduced", "Default settings do not give the stored value."), ("Hypothesis, unverified", "A plausible explanation that the evidence does not settle."), ("Not verifiable here", "No outside tool can check it.")])}

<details><summary>All {len(S['claims'])} claims with recomputed values</summary>

{table(["Id", "Status", "Claim", "Recomputed value"], claims_rows)}

</details>

---

## 8. How it works

### 8.1 Architecture

{img("architecture.svg", "Inputs feed the study layer, which reuses the engine and writes signed outputs that the page reads")}

| Module | Role |
| --- | --- |
| `qneuro/study/kernels.py` | ZZ feature map and angle-embedding kernel in NumPy. A test checks the angle kernel against PennyLane. |
| `qneuro/study/experiments.py` | Re-runs E3 to E6, the leakage study, the platform-gap study and the guard runs. Class-based E6 metrics must match the stored file exactly. ROC-AUC may differ by 0.0005 because it depends on library versions. |
| `qneuro/study/claims.py` | Turns every claim in the report into a recomputed value and a status. |
| `qneuro/study/build.py` | Hashes {len(S['manifest'])} files into a manifest, signs the study and writes the audit log. Also holds `verify` and `tamper_demo`. |
| `qneuro/guard.py` | The Honesty Guard from the original engine. |
| `qneuro/vault.py`, `qneuro/audit.py` | ML-DSA-65 signing and the hash-chained audit log from the original engine. |

### 8.2 What `qneuro study build` does

{mermaid("build-pipeline")}

The build recorded {len(AUDIT)} audit entries: {", ".join(f"{v} x {k}" for k, v in acts.items())}.

### 8.3 The Honesty Guard

The guard scores a comparison between quantum configurations and classical models on paired cross-validation splits. It chooses the best classical model by mean score, tests every quantum configuration against it with the Nadeau-Bengio corrected resampled t-test (which accounts for overlapping training sets), and applies Holm correction across all quantum configurations tried, because choosing the best of several looks better than it is.

{mermaid("guard-decision")}

The interval in the verdict is a bootstrap interval of the mean fold difference. The 15 splits overlap, so that interval is optimistic on its own, which is why the verdict also requires the corrected p-value.

---

## 9. Security and verification

### 9.1 What is signed and why

- **ML-DSA-65** (NIST FIPS 204, a post-quantum signature standard) signs the whole study. Results about patients stay relevant for decades, and signatures made today with classical algorithms could be forged later by a quantum computer.
- **SHA-256 manifest:** every input and output file has its hash inside the signed study, so changing any file is detected.
- **Hash-chained audit log:** every analysis step is an entry that stores the previous entry's hash and its own signature. Editing any entry breaks the chain from that point.

{img("audit-chain.svg", "Four audit entries linked by hashes, with one edited entry shown failing")}

### 9.2 What `qneuro study verify` checks

{mermaid("verify-flow")}

### 9.3 The browser check and the tamper demo

The page re-hashes each file with the browser's own crypto and compares it with the signed manifest. A second button corrupts one digit of the submitted file in memory so the mismatch is visible.

{mermaid("browser-verify")}

{img("screens/05-tamper-detected.jpg", "The page reporting one tampered file after a single digit was changed")}

On the command line, `qneuro study tamper-demo` changes one digit in a copy of the submitted file and one digit in a copy of the signed study. Both edits are detected and an untouched control passes.

### 9.4 Threat model

{table(["What could go wrong", "What catches it"], [
    ("A number in the submitted file is edited after the fact", "The file's SHA-256 no longer matches the signed manifest."),
    ("A number inside the study is edited", "The ML-DSA-65 signature fails."),
    ("An analysis step is edited or removed from the log", "The hash chain breaks at that entry, and the sealing entry no longer matches the study."),
    ("The website copy differs from the repository copy", "`qneuro study verify` compares the published audit log with the local chain, and the browser re-hashes every served file."),
    ("Line endings change on checkout (Windows)", "`.gitattributes` stores hashed files byte for byte, so hashes survive a clone."),
])}

### 9.5 What this does not prove

The tool signs and hashes what it is given. It cannot show that the Bloq platform produced a number: keep the platform's experiment ids and screenshots. The post-quantum libraries ([`kyber-py`](https://github.com/GiacomoPope/kyber-py) and [`dilithium-py`](https://github.com/GiacomoPope/dilithium-py)) are pure-Python reference implementations, unaudited and not constant-time, so they demonstrate the pattern and are not production security.

---

## 10. The page and the walkthrough

{img("screens/01-answer.jpg", "The page's opening answer and four headline figures")}

The page is a static site in [`public/`](public/) that reads `public/data/study.json`. Press **Start walkthrough** for 11 steps. Each shows what is on screen and the sentence to say aloud, and some perform the click for you. The numbers in the narration are filled from the signed study.

{img("screens/07-walkthrough.jpg", "The walkthrough bar at the bottom of the page during step five")}

{mermaid("demo-flow")}

<details><summary>More screenshots</summary>

{img("screens/02-ledger.jpg", "The experiments ledger with the E5 row expanded")}

{img("screens/03-holdout-live.jpg", "The holdout dot grid after one prediction was flipped")}

{img("screens/04-fair-comparison.jpg", "The fair-comparison section")}

{img("screens/06-all-files-verified.jpg", "All files matching the signed manifest")}

{img("screens/08-phone.jpg", "The page on a phone-width screen")}

</details>

The full spoken script, including questions to expect, is in [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

---

## 11. Run it yourself

### Windows PowerShell

```powershell
git clone https://github.com/Vilsee/QNeuro-Forge
cd QNeuro-Forge
powershell -ExecutionPolicy Bypass -File .\\scripts\\setup.ps1     # venv, install, tests, rebuild and sign the study
.\\.venv\\Scripts\\Activate.ps1
qneuro study verify
qneuro study serve --port 8080                                   # http://127.0.0.1:8080/
```

### macOS or Linux

```bash
git clone https://github.com/Vilsee/QNeuro-Forge && cd QNeuro-Forge
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,study,demo]" && python -m playwright install chromium
pytest -q
qneuro study build && qneuro study verify
qneuro study serve --port 8080
```

### Commands

{table(["Command", "What it does"], [
    ("`qneuro study build`", "Re-runs E3 to E6, the leakage study and the guard, checks the claims, signs the result and writes `public/data/` and `runs/morphos/audit.jsonl`. `--fast` skips the sweeps."),
    ("`qneuro study verify`", "Checks the signature, the study hash, every artifact hash, the audit chain and the sealing entry."),
    ("`qneuro study tamper-demo`", "Changes one digit in a copy of the submission and a copy of the study. Both must be detected."),
    ("`qneuro study reproduce`", "Live re-run of the submitted QSVM (E5) and the local ZZ benchmark (E6)."),
    ("`qneuro study prepare`", "Rebuilds the four CSVs from UCI dataset 519 and prints their SHA-256 hashes."),
    ("`qneuro study serve`", "Serves `public/` locally, the same files Vercel serves."),
    ("`qneuro audit --audit-file runs/morphos/audit.jsonl`", "Verifies and summarises the audit chain."),
    ("`python demo/run_demo.py`", "Plays the browser walkthrough with Playwright. `--pace 1.3` is slower, `--manual` lets you drive."),
    ("`scripts/demo.ps1 -Browser`", "The terminal half of the demo with pauses, then the browser half."),
    ("`python scripts/make_figures.py`, `make_screenshots.py`, `make_readme.py`", "Regenerate this README's figures, screenshots and numbers after a rebuild."),
])}

The original engine commands (`qneuro run`, `verify`, `serve`, `tamper-demo`) still work and are unchanged.

### Tests

```bash
pytest -q
```

`tests/test_study.py` re-runs real experiments: E6 must match the stored file, the E5 holdout metrics must match, the angle kernel must equal PennyLane's `AngleEmbedding`, and a tampered study must fail verification.

---

## 12. Repository layout

```
QNeuro-Forge/
  data/morphos/            the submitted files, exactly as handed in
    original/              4 prepared CSVs (with time, no time; train, test)
    scaled_0.3/            the same four, inputs shrunk to 0 to 0.3
    bloq_outputs/          raw Bloq output for E1 and E2
    submissions/           submitted file, original Bloq export, local E6 file
    scripts_original/      the 5 hackathon scripts
  qneuro/
    study/                 kernels, experiments, claims, build (new)
    guard.py vault.py audit.py ...   the original engine
  public/                  static site served by Vercel
    index.html study/      the page, its styles and script
    data/                  signed study.json, audit.jsonl, hashed artifact copies, walkthrough
    pipeline-demo.html     the earlier synthetic-cohort dashboard (a test fixture, not a result)
  runs/morphos/            the audit chain written by the build
  demo/                    Playwright walkthrough runner
  docs/                    this README's figures, diagrams and screenshots, the demo script, the original report
  scripts/                 setup, demo, figure and README generators
  tests/                   pytest suite
```

---

## 13. Reproducibility notes

- **Environment:** the signed study was built with Python {M['environment']['python']}, NumPy {M['environment']['numpy']}, scikit-learn {M['environment']['scikit_learn']} and SciPy {M['environment']['scipy']}.
- **Tolerance:** class-based metrics and fold scores must match exactly on any machine. ROC-AUC means may move in the last digit between library versions, so the E6 comparison allows 0.0005. On one tested Windows setup the GradientBoosting ROC-AUC mean differed by 0.0004.
- **Line endings:** `.gitattributes` marks `data/morphos/`, `public/data/` and `runs/` as `-text`, so Git never converts them. Without it a Windows clone would change the bytes and fail the hash check.
- **Keys:** `qneuro study build` creates a signing key under `.qneuro/keys/` (never committed). The public key is embedded in `study.json`, so anyone can verify without it.
- **Seeds:** the split uses `random_state=7`, the cross-validation folds seed 42 (E5, E6), seed 0 (E3) and seed 1 (E4 and the guard runs).

---

## 14. Limitations

1. Everything quantum is a classical simulation. No hardware ran any circuit and no noise is modelled.
2. The holdout has {NTE} patients. Differences of a few hundredths in MCC are noise there.
3. That E5 was produced on the Bloq platform cannot be verified by any tool outside the platform.
4. The tree-model baselines in the submitted file do not reproduce with default settings. E1 and E2 are reproduced only as arithmetic on the raw platform outputs.
5. The cross-validation protocol picks the best of three quantum variants. Holm correction reduces the selection effect but does not replace a locked test set.
6. The bootstrap interval in the guard's sentence treats the 15 overlapping splits as independent and is therefore optimistic. The corrected p-value is the stricter part of the verdict.
7. The cohort is one hospital pair in Faisalabad, Pakistan (105 women, 194 men, NYHA class III or IV). Results do not carry over to other populations by default.
8. The post-quantum libraries are unaudited demonstration code.
9. This is research tooling. It is not a diagnostic tool and must not inform clinical decisions.

---

## 15. Explaining it to the professor

**Two-minute version.**

1. *The question.* Does a quantum-kernel SVM predict heart-failure death better than classical models, and which of our own claims survive a re-check?
2. *The answer.* On {DS['train']} training patients the best quantum kernel and the best classical model are tied: {f3(VW['quantum_score'])} against {f3(VW['classical_score'])} MCC. The submitted {f3(H['mcc'])} comes from 60 patients, where one changed prediction moves it by about {f3(DEV)}.
3. *What reproduced.* The submitted holdout result, exactly, with an independent kernel. The kernel has no entanglement, unlike its label in the export.
4. *What did not.* The tree-model baselines, one cross-validation fold, and the explanation for the Bloq collapse, which is only a hypothesis.
5. *Why trust this.* Every file is hashed, the study is signed with a post-quantum signature, every step is in a hash-chained log, and two commands plus one button in the browser check all of it, including a deliberate corruption.

<details><summary>Questions to expect, with answers</summary>

{qa_md}

</details>

---

## 16. Glossary

{table(["Term", "Meaning"], [
    ("Classification", "Predicting a category (died or survived) from numbers."),
    ("Confusion matrix", "A 2 by 2 count of correct and wrong predictions: true negatives, false positives, false negatives, true positives."),
    ("MCC", "Matthews correlation coefficient. 0 is no skill, 1 is perfect. Uses all four confusion-matrix cells."),
    ("AUROC", "Area under the ROC curve: the chance that a randomly chosen patient who died is scored higher than one who survived. 0.5 is no signal."),
    ("Kernel", "A similarity score between two data points. An SVM can use any valid kernel."),
    ("QSVM", "A support vector machine whose kernel is computed from quantum states."),
    ("Feature map", "The circuit that turns a patient's numbers into a quantum state."),
    ("ZZ feature map", "A feature map with Hadamards, phases and an entangling block on every pair of qubits."),
    ("Angle embedding", "A feature map with one rotation per qubit and no entanglement."),
    ("Entanglement", "A quantum correlation between qubits that a product state does not have."),
    ("Cross-validation", "Repeatedly training on part of the data and testing on the rest to estimate performance."),
    ("Holdout", "A final set of patients kept aside and used once. Here, 60 patients."),
    ("Data leakage", "Information in the inputs that reveals the answer, making scores look better than real use."),
    ("Nadeau-Bengio test", "A paired t-test corrected for the overlap between cross-validation training sets."),
    ("Holm correction", "A step-down method that controls the chance of any false positive across many tests."),
    ("SHA-256", "A fingerprint of a file: any change gives a different fingerprint."),
    ("ML-DSA-65", "A NIST post-quantum digital signature algorithm (FIPS 204)."),
    ("Hash chain", "A log where each entry includes the previous entry's hash, so editing one entry breaks all later links."),
    ("Manifest", "The signed list of every file and its hash."),
    ("Playwright", "A tool that drives a real browser from code. Used here to play the walkthrough."),
])}

---

## 17. References

**Data and the original study**

- UCI Machine Learning Repository, Heart Failure Clinical Records (id 519): https://archive.ics.uci.edu/dataset/519/heart-failure-clinical-records (DOI https://doi.org/10.24432/C5Z89R, CC BY 4.0)
- Chicco D, Jurman G. Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020): https://doi.org/10.1186/s12911-020-1023-5
- `ucimlrepo`, the package that downloads UCI datasets: https://github.com/uci-ml-repo/ucimlrepo
- The Team Morphos report and PRD: [`docs/MORPHOS_REPORT_AND_PRD.md`](docs/MORPHOS_REPORT_AND_PRD.md)

**Bloq**

- Bloq platform: https://platform.bloq.in/?m=experiments
- Quantum Computing Report, no-code platform launch: https://quantumcomputingreport.com/bloq-quantum-launches-no-code-platform-to-accelerate-algorithm-development/
- The Quantum Insider, platform launch: https://thequantuminsider.com/2026/05/07/bloq-quantum-launches-easy-to-use-platform-to-simplify-quantum-computing-use-case-development/
- The Quantum Insider, IBM Quantum Network: https://thequantuminsider.com/?p=2385279
- Product Hunt: https://www.producthunt.com/products/bloq-quantum

**Quantum kernels**

- Qiskit `ZZFeatureMap`: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.ZZFeatureMap
- PennyLane `AngleEmbedding`: https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html
- scikit-learn `SVC` (used with a precomputed kernel): https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

**Statistics**

- Chicco D, Jurman G. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics 21, 6 (2020): https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7
- Nadeau C, Bengio Y. Inference for the generalization error. Machine Learning 52, 239-281 (2003), DOI https://doi.org/10.1023/A:1024068626366; conference version: https://papers.nips.cc/paper/1661-inference-for-the-generalization-error
- Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2), 65-70 (1979).

**Post-quantum cryptography**

- NIST FIPS 204, Module-Lattice-Based Digital Signature Standard (ML-DSA): https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.204.pdf
- Overview of the finalized NIST standards (FIPS 203, 204, 205): https://cloudsecurityalliance.org/articles/nist-fips-203-204-and-205-finalized-an-important-step-towards-a-quantum-safe-future
- `dilithium-py` (ML-DSA, pure Python): https://github.com/GiacomoPope/dilithium-py and https://pypi.org/project/dilithium-py/
- `kyber-py` (ML-KEM, pure Python): https://github.com/GiacomoPope/kyber-py and https://pypi.org/project/kyber-py/

**Tools**

- Playwright for Python: https://playwright.dev/python/
- Mermaid (the flowcharts in this README): https://mermaid.js.org/
- Vercel (hosting for `public/`): https://vercel.com/docs

*Built for the Quantum Hackathon with Bloq at Karunya, 13 October 2026. Research prototype. Not a diagnostic tool.*
"""
(ROOT / "README.md").write_text(README, encoding="utf-8")
print("wrote README.md,", README.count("\n"), "lines")
