# Security Policy and Cryptographic Notice

## Demonstration Status

QNeuro Forge is an open-source academic research toolkit for benchmarking quantum kernel methods against classical baselines on tabular cohorts. It is not an approved medical device or diagnostic instrument.

## Cryptographic Notice: Unaudited Implementation

The post-quantum cryptographic primitives utilized in this repository:
- **ML-DSA-65**: `dilithium-py`
- **ML-KEM-768**: `kyber-py`
- **Authenticated Encryption**: `cryptography` (AES-256-GCM)

are Python reference implementations intended for educational, experimental, and research demonstration purposes only. Neither `kyber-py` nor `dilithium-py` has received formal independent cryptographic audits or side-channel resistance verification.

**Do not deploy these implementations to protect production clinical systems, raw patient protected health information (PHI), or safety-critical clinical infrastructure.**

## Reporting Vulnerabilities

To report a vulnerability or defect, open an issue in the research repository with details on reproduction steps and environment configuration.
