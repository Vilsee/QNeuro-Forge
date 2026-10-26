# QNeuro Forge

Re-check, sign and verify a quantum-kernel study: every number from Team Morphos's Bloq hackathon entry on heart-failure survival, recomputed by code and sealed with post-quantum signatures.

The repository has two layers.

1. **The study layer** (`qneuro/study`, `public/`, `data/morphos`). It takes the files submitted to the Quantum Hackathon with Bloq, re-runs the experiments, checks every claim in the project report, and publishes a signed ledger as a static site.
2. **The engine** (`qneuro/*.py`). A quantum-kernel versus classical benchmark with a statistical guard, a data gateway, ML-DSA-65 signatures and a hash-chained audit log. The study layer reuses its guard, signing and audit chain.

## The study in one paragraph

Team Morphos built a quantum-kernel SVM (QSVM) to predict death in the UCI Heart Failure Clinical Records dataset (299 patients, ranked by Matthews correlation coefficient, MCC). The submitted file reports a holdout MCC of 0.746 on 60 patients. This repository asks what survives a re-check.

| Question | Answer from the code |
| --- | --- |
| Does the submitted QSVM result reproduce? | Yes. Five class-based holdout metrics match to nine decimals with the same confusion matrix, [[36, 5], [2, 17]]. ROC-AUC differs by 0.0013. |
| Which kernel is it? | An angle-embedding kernel with no entanglement. A true ZZ kernel on the same inputs gives holdout MCC 0.672. |
| Is the quantum model better than classical? | Not demonstrably. Over 15 paired cross-validation splits the best quantum kernel scores 0.600 MCC and the best classical model (XGBoost) 0.599. Guard verdict: `NO_DIFFERENCE`. |
| How reliable is the 60-patient holdout? | One changed prediction moves MCC between 0.702 and 0.790. Two move it between 0.659 and 0.834. |
| Does the follow-up `time` feature matter? | Alone it scores AUROC 0.812. Replacing it with serum sodium lowers MCC for every model by 0.164 to 0.229. |
| Why did the Bloq runs E1 and E2 collapse? | Unknown. Local ZZ kernels on inputs in 0 to 1 never predicted "no deaths" in 300 random splits, so split luck does not explain it. Kernel concentration at wide input ranges is a hypothesis. |
| What was edited before submission? | Three text fields, listed in the page. Every number is identical. |

Of 20 claims in the report, 12 reproduce, 4 are supported by local analysis, and 4 do not fully hold: one partly reproduced (one cross-validation fold), one not reproduced (the tree-model baselines), one hypothesis, and one not verifiable (that E5 ran on Bloq, which no outside tool can check).

The guard's own sentence for the with-time comparison:

> Observed MCC delta of +0.0009 between qsvm_zz_x0.3 (0.5999) and xgboost (0.5990) with bootstrap 95% CI [-0.0611, +0.0653] and Nadeau-Bengio corrected p = 1.0000 across 3 quantum configurations does not reject the null hypothesis at threshold 0.02. No conclusive advantage demonstrated.

All figures come from `public/data/study.json`. Rebuild them with `qneuro study build`.

## Quickstart on Windows PowerShell

```powershell
git clone https://github.com/Vilsee/QNeuro-Forge
cd QNeuro-Forge
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1      # venv, install, tests, rebuild and sign the study
.\.venv\Scripts\Activate.ps1
qneuro study verify                                               # signature, 18 file hashes, audit chain
qneuro study serve --port 8080                                    # study site at http://127.0.0.1:8080/
```

Presenting: `.\scripts\demo.ps1` runs the terminal half with pauses, and `python demo\run_demo.py` plays the browser walkthrough with Playwright. The full script with spoken lines is `docs/DEMO_SCRIPT.md`.

## Commands

| Command | What it does |
| --- | --- |
| `qneuro study build` | Re-runs E3 to E6, the leakage study and the guard, checks 20 claims, signs the result and writes `public/data/` and `runs/morphos/audit.jsonl`. `--fast` skips the sweeps. |
| `qneuro study verify` | Checks the ML-DSA-65 signature, the study hash, every artifact hash, the audit chain and the sealing entry. |
| `qneuro study tamper-demo` | Changes one digit in a copy of the submission and in a copy of the study. Both edits must be detected. |
| `qneuro study reproduce` | Live re-run of the submitted QSVM (E5) and the local ZZ benchmark (E6). |
| `qneuro study prepare` | Rebuilds the four train and test CSVs from UCI dataset 519 and prints their SHA-256 hashes. |
| `qneuro study serve` | Serves `public/` locally, the same files Vercel serves. |
| `qneuro run`, `verify`, `serve`, `tamper-demo`, `audit` | The original engine commands. |

## How the study layer works

- `data/morphos/` holds the submitted files exactly as handed in: the four prepared CSVs, the raw Bloq outputs for E1 and E2, the submitted file, the original Bloq export, the local E6 file and the five scripts.
- `qneuro/study/kernels.py` implements the ZZ feature map and the angle-embedding kernel in NumPy. A test checks the angle kernel against PennyLane.
- `qneuro/study/experiments.py` re-runs the experiments. E6 reproduces the stored file: every fold MCC and class-based metric is identical, and ROC-AUC agrees to within 0.0005 (it can move in the last digit between library versions).
- `qneuro/study/claims.py` turns every claim in the report into a recomputed value and a status: reproduced, partly reproduced, not reproduced, supported, hypothesis, or not verifiable.
- `qneuro/study/build.py` hashes 18 files into a manifest, signs the study with ML-DSA-65 and records each step in a signed, hash-chained audit log. `qneuro.guard` decides the quantum-versus-classical verdict, and it is the only code that words an advantage claim.
- `public/index.html` renders the signed study. The browser recomputes every file hash itself. It cannot check an ML-DSA signature, so `qneuro study verify` does that.

## The page

`public/` is a static site deployed to Vercel. The page has a built-in walkthrough: the **Start walkthrough** button shows the narration to say aloud for each of 11 steps and, where useful, performs the click for you. `public/pipeline-demo.html` is the earlier synthetic-cohort dashboard, kept as a software test fixture. It is not a result.

## Limitations

1. Everything quantum here is a classical simulation. No hardware ran any circuit, and no noise is modelled.
2. The holdout has 60 patients. Differences of a few hundredths in MCC are noise there.
3. That E5 was produced on the Bloq platform cannot be verified by any tool outside the platform. Keep the experiment ids and screenshots.
4. The tree-model baselines in the submitted file do not reproduce with default settings. E1 and E2 are reproduced only as arithmetic on the raw platform outputs.
5. The cross-validation protocol picks the best of three quantum variants. Holm correction reduces the selection effect but does not replace a locked test set.
6. The cohort is one hospital pair in Faisalabad, Pakistan (105 women, 194 men, NYHA class III or IV). Results do not carry over to other populations by default.
7. `kyber-py` and `dilithium-py` are pure-Python reference implementations, unaudited and not constant-time. They demonstrate the pattern and are not production security.
8. This is research tooling. It is not a diagnostic tool and must not inform clinical decisions.

## Data and licences

Dataset: Heart Failure Clinical Records, UCI Machine Learning Repository, id 519, DOI 10.24432/C5Z89R, licensed CC BY 4.0. Introductory paper: Chicco and Jurman (2020), BMC Medical Informatics and Decision Making 20, 16, https://doi.org/10.1186/s12911-020-1023-5.

## Tests

```powershell
pytest -q
```

58 tests. `tests/test_study.py` re-runs real experiments: E6 must match the stored file, E5 holdout metrics must match, the product kernel must equal PennyLane's angle embedding, and a tampered study must fail verification.
