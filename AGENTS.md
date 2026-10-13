# AGENTS.md — Agent System Specification & Guardrails

## Purpose
QNeuro Forge is an open-source Python research toolkit evaluating quantum-kernel versus classical machine learning baselines on tabular neurodevelopmental cohorts.

## Core Directives for Autonomous Agents

1. **Non-Negotiable Empirical Truth**
   - Never hardcode, mock, invent, or edit any benchmark number, metric, or p-value.
   - All presented values must originate from verifiable executions written to `runs/`.

2. **Synthetic Data Boundaries**
   - All synthetic datasets must be labelled `SYNTHETIC` in metadata, outputs, and UI.
   - Generative algorithms must be symmetric and never tuned to structurally favor any model.

3. **Sole Advantage Arbiter**
   - Only `qneuro.guard` is authorized to formulate or decide a "quantum advantage" claim.
   - Claims require conservative thresholds (e.g. delta > 0.02) and must report confidence intervals and corrected p-values.

4. **Gateway Privacy & Data Isolation**
   - The experiment agent must never receive or inspect raw data rows.
   - Only structural schemas, row counts, class balances, and per-feature aggregate statistics are exposed through `qneuro.gateway`.

5. **Post-Quantum Cryptography**
   - Signing: ML-DSA-65 (`dilithium-py`).
   - Key encapsulation: ML-KEM-768 (`kyber-py`).
   - Data encryption: AES-256-GCM (`cryptography`).
   - Never write custom crypto primitives. Keys sourced from environment variables only.

6. **Code Standards**
   - Python 3.11+, full type annotations, Pydantic v2 models at every interface boundary.
   - No banner comments, no emoji, no leftover TODOs, no unverified claims.
