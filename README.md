# QNeuro Forge

QNeuro Forge is an open-source research toolkit evaluating quantum-kernel support vector classifiers against classical machine learning baselines on small tabular neurodevelopmental cohorts. An autonomous experiment agent explores candidate quantum configurations while sequestered behind a data gateway that strictly isolates raw patient rows. Every empirical benchmark result is notarized with post-quantum digital signatures (ML-DSA-65) and logged in an append-only, tamper-evident hash chain.

## Empirical Benchmark Results

The following unedited results originate from an autonomous benchmark run (`run_id: c097482a2540`) evaluated on a 120-sample `SYNTHETIC` neurodevelopmental cohort across 15 stratified folds (5 splits x 3 repeats):

| Model Configuration | AUROC | 95% Confidence Interval | F1 Score | Wall-Clock Time |
| :--- | :--- | :--- | :--- | :--- |
| `quantum_svc_angle_q6_r1_C1.0` | 0.5633 | [0.5183, 0.6083] | 0.6564 | 2.56s |
| `quantum_svc_iqp_q4_r2_C1.0` | 0.5571 | [0.4683, 0.6460] | 0.6080 | 9.12s |
| `quantum_svc_iqp_q6_r2_C1.0` | 0.5487 | [0.4754, 0.6219] | 0.6417 | 17.64s |
| `logistic_regression_C1.0` | 0.5432 | [0.4744, 0.6119] | 0.6289 | 1.03s |
| `quantum_svc_angle_q4_r1_C1.0` | 0.5409 | [0.5119, 0.5698] | 0.6748 | 2.18s |
| `xgboost_C1.0` | 0.5170 | [0.4432, 0.5909] | 0.5558 | 17.36s |
| `rbf_svc_C1.0` | 0.4609 | [0.3949, 0.5269] | 0.5901 | 2.00s |

### Arbitration Ruling (`qneuro/guard.py`)

> Observed AUROC delta of +0.0201 between `quantum_svc_angle_q6_r1_C1.0` (0.5633) and `logistic_regression_C1.0` (0.5432) with bootstrap 95% CI [-0.0518, +0.0976] and Nadeau-Bengio corrected p = 1.0000 across 4 quantum configurations does not reject the null hypothesis at threshold 0.02. No conclusive advantage demonstrated.

- Verdict Category: `NO_DIFFERENCE`
- Digital Signature: `ML-DSA-65` (FIPS 204)
- Public Key Fingerprint: `SHA256:d112:cb8c:81e9:a4f8:f1e6:6ffd:6c6e:ba04`
- Audit Ledger Status: 24 entries linked and verified

---

## Limitations

1. **Simulator-Only and Low Qubit Widths**: Quantum circuits are simulated numerically via PennyLane statevector representation on classical CPU registers restricted to 4 to 6 qubits. The system does not run on real physical quantum processing units (QPUs) and does not model physical hardware noise, gate infidelities, or decoherence.
2. **Synthetic Data**: Default cohorts are synthetically generated to test system pipelines and methodology. Synthetic datasets cannot reproduce the multi-factorial clinical heterogeneity of real-world patient cohorts.
3. **Agent Selection Effect**: Although the guard module applies a step-down Holm-Bonferroni correction across all quantum configurations proposed by the agent, repeated evaluations across repeated folds without a permanently locked, holdout clinical test partition retain risk of optimistic estimation.
4. **Unaudited Educational Cryptography**: Post-quantum algorithms (`kyber-py` and `dilithium-py`) are pure-Python reference implementations intended for demonstration. They lack independent formal audits, constant-time guarantees, and side-channel resistance.
5. **Not a Diagnostic Tool**: QNeuro Forge is an open-source exploratory methodology toolkit. It is not an approved medical device or software as a medical device (SaMD) and must not be used for clinical decision-making or medical diagnosis.

---

## Commands to Reproduce

### 1. Installation

```bash
git clone https://github.com/Vilsee/QNeuro-Forge.git
cd QNeuro-Forge
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Execute Benchmark Pipeline

Run the autonomous agent benchmark across candidate quantum circuits and classical baselines using the deterministic offline planner:

```bash
qneuro run --budget 4 --samples 120 --seed 42
```

### 3. Verify Cryptographic Integrity

Verify the ML-DSA-65 signature on the generated artifact:

```bash
qneuro verify runs/c097482a2540/report.json
```

Verify append-only audit ledger continuity:

```bash
qneuro audit
```

Demonstrate tamper detection by altering a single byte:

```bash
qneuro tamper-demo
```

### 4. Launch Web Dashboard

Serve the notarised instrument ledger at `http://127.0.0.1:8000`:

```bash
qneuro serve
```

### 5. Run Test Suite

```bash
pytest -v
```
