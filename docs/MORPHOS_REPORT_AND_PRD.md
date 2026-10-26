# Project Report & PRD: Quantum-Kernel (QSVM) Survival Prediction on Heart Failure Clinical Records

Team Morphos · Hackathon track M1 (Heart Failure Clinical Records Survival Prediction) · Ranking metric: MCC · Experiments run 13 October 2026 (UTC)

## 1. Executive summary

We built a quantum-kernel support vector machine (QSVM) pipeline to predict in-hospital-cohort death events from the UCI Heart Failure Clinical Records dataset (299 patients, target `death_event`). The work covered data preparation (two four-feature variants), two runs on the Bloq platform, a local statevector simulation study to understand why the platform runs scored poorly, a submission file exported from the Bloq pipeline, and a corrected, honestly labelled version of that file, which is the one that was submitted.

Key outcomes:

- The two raw Bloq config runs both collapsed toward the majority class (MCC 0.000 and 0.012), so neither was submittable.
- The submitted file reports a QSVM holdout MCC of 0.746 (accuracy 0.883, recall 0.895, ROC-AUC 0.933) on a 60-patient test set, comparable to the best classical models in the same file (RandomForest, GradientBoosting and SVM-RBF at 0.778).
- A separate local true-ZZ simulation reached holdout MCC 0.806 with features rescaled to the 0 to 0.3 range, but it was not run on Bloq, so it was not submitted.
- Main risks: the outcome-leaking `time` feature (allowed by the track), a very small test set, and a gap between platform and local results that we could not fully explain.

## 2. Product Requirements Document (PRD)

### 2.1 Problem statement

Heart failure outcomes are hard to predict from a handful of clinical measurements. The hackathon asks teams to build a binary classifier for `death_event` using a quantum kernel method on the Bloq platform, ranked by Matthews Correlation Coefficient (MCC), which is robust to the roughly 68/32 class imbalance.

### 2.2 Objectives

1. Produce a QSVM on 4 qubits (one feature per qubit) with a strong holdout MCC.
2. Benchmark it against classical baselines on identical data so the claim is defensible.
3. Deliver a submission JSON in the track's format, named `T<team>_<run>.json`, before the 4:45 PM deadline.
4. Report results honestly, including uncertainty and leakage risk.

### 2.3 Non-goals

- Clinical deployment or regulatory-grade validation.
- Real quantum hardware execution (simulator only).
- Beating all classical models; the goal was a competitive, honest quantum result.

### 2.4 Stakeholders

| Role | Interest |
| --- | --- |
| Team Morphos (author) | Competitive score, clean documentation |
| Hackathon judges | Valid Bloq-pipeline result, MCC, honest reporting |
| Organizers | Correct filename and JSON schema |

### 2.5 Functional requirements

| ID | Requirement | Status |
| --- | --- | --- |
| FR1 | Prepare train/test data with a single shared patient split | Done (`prepare_hf.py`) |
| FR2 | Provide a no-`time` and a with-`time` feature variant | Done |
| FR3 | Run QSVM on Bloq with 4 qubits, depth 1 | Done (two config runs plus submission export) |
| FR4 | Report CV and holdout MCC, accuracy, F1, precision, recall, ROC-AUC | Done (submission JSON) |
| FR5 | Compare against classical baselines with confidence intervals and paired test | Done |
| FR6 | Follow naming `T<team>_<run>.json` | Pending team number (team name: Morphos) |

### 2.6 Non-functional requirements

- Reproducibility: fixed random seeds and a train-only scaler fit.
- Honesty: metadata must describe the model that actually ran.
- Leakage awareness: document that `time` can carry outcome information.

### 2.7 Success metrics

| Metric | Target | Achieved (submitted file) |
| --- | --- | --- |
| Holdout MCC | Above 0.5 | 0.746 |
| Cross-validated MCC | Above 0.4 | 0.507 |
| Competitive with best classical | Within about 0.05 holdout MCC | 0.746 vs 0.778 |
| Valid Bloq-pipeline run | Required by judges | Yes (submitted file) |

### 2.8 Constraints and assumptions

- Only 4 features may be used (4 qubits).
- The platform's own settings are limited: qubits, depth, device, feature map, NumPy inner product, train/test ratio, dataset and selected features.
- Judges require a Bloq-run result; locally computed results are not accepted.
- The `time` feature is allowed by the track.
- Assumption: the 60-row test split is representative; with so few rows, one or two patients move MCC noticeably.

## 3. Data

### 3.1 Source

UCI Heart Failure Clinical Records dataset (UCI repository id 519, fetched via `ucimlrepo` or a local CSV named `heart_failure_clinical_records_dataset.csv`). 299 patients, target `death_event` (1 = death during follow-up).

### 3.2 Split and scaling

| Item | Value |
| --- | --- |
| Split | Stratified 80/20 on `death_event`, `random_state=7` |
| Train rows | 239 (death rate 0.322) |
| Test rows | 60 (death rate 0.317, about 19 deaths) |
| Scaler | `MinMaxScaler` fit on train only, applied to train and test |
| Test range note | Test values fall slightly outside \[0,1\] (min about -0.094, max about 1.047) because the scaler never saw test data |

### 3.3 Feature variants

| Variant | Features | Concern |
| --- | --- | --- |
| `hf_notime` | ejection\_fraction, serum\_creatinine, age, serum\_sodium | Deployable at prediction time |
| `hf_withtime` | ejection\_fraction, serum\_creatinine, age, time | `time` is follow-up duration; patients who die have shorter follow-up, so it leaks the outcome |

The CSV files have the four scaled feature columns followed by `death_event`. The shrunk copies (`*_s03_*`) multiply every feature by 0.3 after clipping test values to \[0,1\].

## 4. Method

### 4.1 Quantum kernel SVM

A QSVM computes a kernel matrix K(a, b) = |⟨φ(a)|φ(b)⟩|² from quantum feature states φ(x), then trains a classical SVM (`SVC(kernel='precomputed')`) on that matrix.

- **Bloq platform runs:** 4 qubits, depth 1, `aer_simulator` (IBM provider, simulator, method automatic), ZZ Feature Map (default), NumPy inner product on.
- **Submitted export:** metadata describes an angle-embedding product-state fidelity kernel at depth 1 on the Bloq cluster simulator. This has no entanglement and is therefore not a true ZZ map (corrected in the label, see 6.2).
- **Local true-ZZ simulation:** exact NumPy statevector, Hadamard on all qubits, phase P(2x\_i), then pairwise parity phases 2(π−x\_i)(π−x\_j) on all 6 qubit pairs (full entanglement), depth 1.
- **`local_qsvm.py`:** a PennyLane angle-embedding kernel (inputs multiplied by π). It is a different kernel from the platform's ZZ map, so its scores are not comparable.

### 4.2 Evaluation protocol

- Cross-validation: 5-fold stratified on the 239 training rows (folds of 48,48,48,48,47).
- Holdout: model trained on all 239 rows, evaluated once on the 60 test rows.
- Metrics: MCC (ranking), accuracy, F1, precision, recall, ROC-AUC.
- 95% confidence interval of fold MCC: t-interval over the 5 folds. Paired t-test of QSVM vs RandomForest on identical folds.

## 5. Experiment log

### E1: Bloq run on `hf_notime_train.csv` (11:11:50 to 11:12:01 UTC, 13 Oct)

Settings: 4 qubits, depth 1, aer\_simulator, ZZ Feature Map, NumPy inner product, train/test ratio 80 (platform re-splits the uploaded 239 rows, giving 48 test rows: 33 class 0, 15 class 1). Features: ejection\_fraction, serum\_creatinine, age, serum\_sodium.

| Metric | Value |
| --- | --- |
| Accuracy | 0.6875 |
| MCC | 0.000 |
| Class 0 precision / recall / F1 | 0.688 / 1.000 / 0.815 |
| Class 1 precision / recall / F1 | 0 / 0 / 0 |
| Macro F1 | 0.407 |
| Confusion matrix \[\[TN, FP\],\[FN, TP\]\] | \[\[33, 0\],\[15, 0\]\] |

Interpretation: a degenerate model that predicted "survived" for every patient; accuracy equals the majority-class baseline (33/48).

### E2: Bloq run on `hf_withtime_train.csv` (11:13:28 to 11:13:39 UTC, 13 Oct)

Same settings as E1, features: ejection\_fraction, serum\_creatinine, age, time. The run was titled "notime" despite using the `time` feature (a labelling mistake).

| Metric | Value |
| --- | --- |
| Accuracy | 0.6667 |
| MCC | 0.0116 |
| Class 1 recall | 0.067 (1 of 15) |
| Macro F1 | 0.453 |
| Confusion matrix | \[\[31, 2\],\[14, 1\]\] |

Interpretation: only marginally different from E1; effectively noise.

### E3: Local simulation sweep 1, depth and input range (true ZZ kernel, SVC default C)

Train on the 239 rows, 5-fold CV, holdout on the 60 test rows. Approximate local results:

| Data | Range | Depth | CV MCC | Test MCC |
| --- | --- | --- | --- | --- |
| notime | 0 to 1 | 1 | 0.34 | 0.40 |
| withtime | 0 to 1 | 1 | 0.52 | 0.67 |
| notime | 0 to 1 | 2 | 0.32 | 0.27 |
| withtime | 0 to 1 | 2 | 0.41 | 0.57 |
| notime | 0 to π | 1 | 0.24 | 0.32 |
| withtime | 0 to π | 1 | 0.16 | 0.23 |
| notime | 0 to 2π | 1 | -0.01 | 0.13 |
| withtime | 0 to 2π | 1 | -0.01 | 0.01 |

Finding: larger input ranges and deeper circuits hurt. The kernel concentrates (all entries become similar) when angles are large, which pushes the SVM toward a constant prediction. This matched the collapse seen in E1 and E2.

### E4: Local simulation sweep 2, preprocessing and range (repeated 5-fold CV, 3 repeats)

Tested minmax, quantile-uniform and 5-95 percentile-clipped inputs at scales 0.3, 0.5, 0.75, 1.0, 1.5.

| Data | Range 0 to 1 (CV / test MCC) | Range 0 to 0.3 (CV / test MCC) |
| --- | --- | --- |
| withtime | 0.54 / 0.67 | **0.60 / 0.81** |
| notime | 0.41 / 0.40 | 0.40 / 0.45 |

Finding: shrinking the range to 0.3 helped withtime; the notime change was within noise. Quantile transforms did not help; clipping gave a small, inconsistent gain on notime. Output: the `*_s03_*` CSV files.

### E5: Bloq pipeline export `1791891125080_m1_heart_failure_submission.json` (11:30:17 UTC)

Track format with QSVM and five classical baselines. Features: ejection\_fraction, serum\_creatinine, age, time. Results below.

| Model | CV MCC (mean) | CV MCC 95% CI | Holdout MCC |
| --- | --- | --- | --- |
| QSVM (quantum kernel) | 0.5068 | \[0.296, 0.717\] | 0.746 |
| LogisticRegression | 0.000 | \[0, 0\] | 0.191 |
| RandomForest | 0.629 | \[0.451, 0.807\] | 0.778 |
| GradientBoosting | 0.575 | \[0.460, 0.690\] | 0.778 |
| SVM-RBF | 0.525 | \[0.334, 0.716\] | 0.778 |
| XGBoost | 0.611 | \[0.516, 0.706\] | 0.656 |

QSVM details: CV accuracy 0.795, CV F1 0.576, CV ROC-AUC 0.870. Fold MCCs 0.318, 0.359, 0.514, 0.633, 0.711.

QSVM holdout: MCC 0.746, accuracy 0.883, F1 0.829, precision 0.773, recall 0.895, ROC-AUC 0.933. Reconstructed confusion matrix on 60 patients: 36 true negatives, 5 false positives, 2 false negatives, 17 true positives.

Statistical comparison vs RandomForest: delta MCC -0.122, 95% CI \[-0.290, 0.045\], paired t = -2.03, p = 0.112 (not significant). Logistic regression predicted the majority class in every fold, so beating it is a weak claim.

### E6: Local true-ZZ benchmark, all models recomputed on identical folds (`m1_heart_failure_submission_zz.json`)

True ZZ map, depth 1, inputs 0 to 0.3 (range chosen by CV MCC: 1.0 gave CV 0.554 / holdout 0.672, 0.3 gave CV 0.569 / holdout 0.806). 5-fold stratified, seed 42.

| Model | CV MCC | Holdout MCC |
| --- | --- | --- |
| QSVM (ZZ) | 0.569 | 0.806 (accuracy 0.90) |
| LogisticRegression | 0.548 | 0.746 |
| RandomForest | 0.587 | 0.681 |
| GradientBoosting | 0.588 | 0.733 |
| SVM-RBF | 0.525 | 0.778 |
| XGBoost | 0.578 | 0.632 |

QSVM vs RandomForest: delta MCC -0.018, CI \[-0.133, 0.097\], t = -0.44, p = 0.68. The file states in `hardware_platform` that it was a local simulation, not a Bloq run. It was not submitted because the judges require a Bloq-run result.

## 6. Final submission

### 6.1 What was submitted

`m1_heart_failure_submission_corrected.json`, derived from the Bloq export in E5, renamed to the required `T<team>_<run>.json` pattern (team name Morphos; team number to be confirmed with the organizers).

### 6.2 Corrections made to the original export

Only three text fields were changed; every metric was verified unchanged.

| Field | Before | After |
| --- | --- | --- |
| `feature_map` | "ZZFeatureMap (Angle Embedding product state fidelity)" | "Angle Embedding fidelity kernel (product-state, no entanglement)" |
| `leakage_audit_note` | Claimed time inflates MCC across quantum and classical baselines (no such comparison in the file) | States that time is permitted but can carry outcome information and that no no-time comparison is included |
| `verdict` | "Follows regulatory guidelines" | QSVM holdout MCC is comparable to classical models; CV MCC is lower than RandomForest's but not significantly (p = 0.112) |

## 7. Decisions and rationale

| Decision | Rationale |
| --- | --- |
| Keep MinMax \[0,1\] data for the submitted pipeline run | Best-tested range for the exported pipeline; larger ranges collapse the kernel |
| Use depth 1 | Depth 2 was worse in every local test |
| Include `time` | Track allows it and it raises MCC strongly (withtime CV about 0.52 vs notime about 0.34 in local tests) |
| Relabel the feature map accurately instead of calling it ZZ | The kernel had no entanglement; a false label would misrepresent the work |
| Do not submit the local ZZ file | Judges require a Bloq-run result; a locally computed file risks rejection |
| Do not edit metrics by hand | Results must come from real runs |

## 8. Risks, limitations and threats to validity

1. **Target leakage:** `time` is follow-up duration and is unknown at true prediction time. The reported MCC is likely higher than for a deployable model. No no-time comparison exists inside the submitted file.
2. **Small test set:** 60 rows with about 19 positives; one or two patients shift holdout MCC substantially, so 0.746 versus 0.778 is not a meaningful gap.
3. **Wide cross-validation uncertainty:** the QSVM CV MCC interval is \[0.30, 0.72\].
4. **Platform vs local gap:** the raw Bloq config runs (MCC about 0) scored far below the local simulation on the same data; the cause (platform defaults, internal rescaling or SVM settings) could not be determined because platform internals are not visible.
5. **Weak baseline:** the logistic regression in the submitted file predicted the majority class in every fold.
6. **Local simulation approximations:** the local ZZ results assume the standard ZZ construction and are directional, not exact predictors of the platform.
7. **Unrecorded scaling:** the exported submission does not record the input scaling used; reproduction of its exact numbers is not guaranteed.
8. **Naming:** the team number was not known at documentation time; the filename must follow `T<team>_<run>.json`.

## 9. Future work

- Run the 0.3-scaled withtime data through Bloq with the ZZ Feature Map and compare against 0.746.
- Add a no-`time` Bloq run to quantify leakage honestly in the same file format.
- Tune the SVM `C` value and class weighting; test alternative feature maps (Pauli, Z) and feature subsets.
- Use repeated or nested cross-validation to tighten uncertainty on a dataset of this size.
- Investigate the platform vs local gap by logging the platform's kernel matrix.

## Appendix A: File inventory

| File | Type | Purpose |
| --- | --- | --- |
| `prepare_hf.py` | Script | Fetches data, stratified split (seed 7), MinMax scaling (train-fit), writes the four CSVs, prints SHA-256 hashes and a classical RBF-SVM MCC reference |
| `local_qsvm.py` | Script | PennyLane angle-embedding kernel SVM (not ZZ); local reference only |
| `hf_notime_train.csv`, `hf_notime_test.csv` | Data | 239 / 60 rows, features EF, creatinine, age, sodium, plus `death_event` |
| `hf_withtime_train.csv`, `hf_withtime_test.csv` | Data | 239 / 60 rows, features EF, creatinine, age, time, plus `death_event` |
| `hf_notime_s03_*.csv`, `hf_withtime_s03_*.csv` | Data | Same data, test clipped to \[0,1\], all features multiplied by 0.3 (four files) |
| `qneuro_hf_qsvm_notime-config.json` | Bloq output | E1 raw platform run (notime), MCC 0 |
| `qneuro_hf_qsvm_notime-config (1).json` | Bloq output | E2 raw platform run (actually withtime), MCC 0.012 |
| `1791891125080_m1_heart_failure_submission.json` | Bloq export | E5 original track-format export |
| `m1_heart_failure_submission_corrected.json` | Submission | E5 with three corrected text fields; the submitted file |
| `m1_heart_failure_submission_zz.json` | Local result | E6 true-ZZ local benchmark; not submitted |
| `sweep1_depth_scale.py`, `sweep2_preprocessing.py`, `zz_benchmark.py` | Scripts | Local simulation sweeps and the E6 benchmark |

## Appendix B: Reproduction steps

1. Place `heart_failure_clinical_records_dataset.csv` next to the scripts (or install `ucimlrepo`), then run `python prepare_hf.py` to regenerate the four CSVs and print their SHA-256 hashes.
2. Upload a `_train.csv` file to Bloq, choose QSVM, 4 qubits, depth 1, ZZ Feature Map, aer\_simulator, NumPy inner product on, train/test ratio 80, target `death_event`, then export the config JSON.
3. Run `python sweep1_depth_scale.py` and `python sweep2_preprocessing.py` for the E3 and E4 tables (requires numpy, pandas, scikit-learn).
4. Run `python zz_benchmark.py` (also needs scipy and xgboost) to regenerate E6.

## Appendix C: Bloq platform settings used

| Setting | Value |
| --- | --- |
| Experiment type | ml |
| Algorithm | QSVM |
| No. of qubits | 4 |
| Circuit depth | 1 |
| Device | aer\_simulator (IBM provider, simulator, method automatic) |
| Feature map | ZZ Feature Map (default) |
| Use NumPy inner product | true |
| Dataset size / train-test ratio | 239 rows / 80 |
| Target / task | death\_event / Binary Classification |

## Appendix D: Glossary

- **MCC (Matthews Correlation Coefficient):** correlation between predictions and truth that stays near 0 for do-nothing models; 1 is perfect, 0 is chance.
- **QSVM:** SVM that uses a quantum-computed kernel (state overlap) instead of a classical kernel.
- **ZZ feature map:** circuit that encodes features with single-qubit rotations plus pairwise entangling phases.
- **Angle embedding:** encodes each feature as a single-qubit rotation with no entanglement (a product state).
- **Data leakage:** using information at training time that would not be available when predicting on new patients.
- **Holdout:** final test set evaluated once after all choices are made.
