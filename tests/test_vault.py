"""Tests for post-quantum signing, key encapsulation, and encryption."""
from __future__ import annotations

import datetime
import os
from pathlib import Path
import pytest
from cryptography.exceptions import InvalidTag

from qneuro.models.schemas import BenchmarkRun, ModelResult
from qneuro.types import ExperimentConfig, FoldScores, GuardVerdict, SignedReport, VerdictCategory
from qneuro.vault import (
    canonical_json,
    decrypt_blob,
    decrypt_file,
    decrypt_payload,
    encapsulate,
    encrypt_blob,
    encrypt_file,
    encrypt_payload,
    get_or_create_dsa_keys,
    get_or_create_kem_keys,
    get_public_key_fingerprint,
    sign_data,
    sign_run,
    verify_run,
    verify_signature,
)


def _sample_run() -> BenchmarkRun:
    return BenchmarkRun(
        run_id="vault-test",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        dataset_name="test",
        dataset_source="SYNTHETIC",
        n_samples=20,
        n_features=4,
        classical_results=[
            ModelResult(
                model_name="LR",
                accuracy=0.75,
                f1_score=0.74,
                precision=0.73,
                recall=0.76,
                train_time_seconds=0.01,
            )
        ],
        quantum_results=[
            ModelResult(
                model_name="QK",
                accuracy=0.72,
                f1_score=0.71,
                precision=0.70,
                recall=0.73,
                train_time_seconds=0.05,
            )
        ],
        random_seed=99,
        python_version="3.11.0",
    )


def _sample_report() -> SignedReport:
    sk, pk = get_or_create_dsa_keys()
    cfg = ExperimentConfig(model_family="quantum_svc", feature_map="angle", n_qubits=4, reps=1, C=1.0)
    fs = FoldScores(
        config_id=cfg.config_id,
        metric="auroc",
        scores=[0.6, 0.62, 0.65, 0.58, 0.61],
        mean_score=0.612,
        std_score=0.025,
        ci_lower=0.58,
        ci_upper=0.64,
    )
    verdict = GuardVerdict(
        category=VerdictCategory.NO_DIFFERENCE,
        metric_used="auroc",
        best_quantum_model=cfg.config_id,
        best_classical_model="baseline_lr",
        quantum_score=0.612,
        classical_score=0.610,
        delta=0.002,
        ci_lower=-0.02,
        ci_upper=0.02,
        p_value=0.85,
        configs_tried=1,
        sentence="No conclusive advantage demonstrated.",
    )
    raw_dict = {
        "report_id": "rep-test-001",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_fingerprint": "abc12345",
        "dataset_source": "SYNTHETIC",
        "configs": [cfg.model_dump()],
        "fold_scores": [fs.model_dump()],
        "wall_clock_seconds": {cfg.config_id: 1.5},
        "verdict": verdict.model_dump(),
    }
    sig_hex, pk_hex = sign_data(raw_dict, sk=sk, pk=pk)
    return SignedReport(
        report_id=raw_dict["report_id"],
        timestamp=datetime.datetime.fromisoformat(raw_dict["timestamp"]),
        dataset_fingerprint=raw_dict["dataset_fingerprint"],
        dataset_source=raw_dict["dataset_source"],
        configs=[cfg],
        fold_scores=[fs],
        wall_clock_seconds=raw_dict["wall_clock_seconds"],
        verdict=verdict,
        signature_hex=sig_hex,
        public_key_hex=pk_hex,
    )


def test_sign_and_verify() -> None:
    run = _sample_run()
    signed = sign_run(run)
    assert verify_run(signed)
    fp = get_public_key_fingerprint(signed.public_key_hex)
    assert fp.startswith("SHA256:")


def test_tampered_run_fails_verification() -> None:
    run = _sample_run()
    signed = sign_run(run)
    signed.run.run_id = "tampered"
    assert not verify_run(signed)


def test_signed_report_verifies_and_byte_flip_fails() -> None:
    """Valid report verifies; flipping a single byte in the serialized payload fails verification."""
    report = _sample_report()
    payload = canonical_json(report.model_dump(exclude={"signature_hex", "public_key_hex", "algorithm"}))
    
    # Check valid verification
    assert verify_signature(payload, report.signature_hex, report.public_key_hex)

    # Flip one byte of the payload
    tampered_bytes = bytearray(payload)
    tampered_bytes[10] ^= 0x01
    assert not verify_signature(bytes(tampered_bytes), report.signature_hex, report.public_key_hex)


def test_decrypt_with_wrong_key_fails() -> None:
    """Decapsulating or decrypting with an incorrect key raises an error."""
    dk_legit, ek_legit = get_or_create_kem_keys()
    
    # Generate an independent other KEM keypair
    from kyber_py.ml_kem import ML_KEM_768
    ek_other, dk_other = ML_KEM_768.keygen()

    plaintext = b"Sensitive neurodevelopmental tabular records"
    encrypted = encrypt_payload(plaintext, kem_ek=ek_legit)

    # Decrypting with legitimate key succeeds
    assert decrypt_payload(encrypted, kem_dk=dk_legit) == plaintext

    # Decrypting with wrong key fails (either ValueError or InvalidTag)
    with pytest.raises((ValueError, InvalidTag, Exception)):
        decrypt_payload(encrypted, kem_dk=dk_other)


def test_file_encryption_roundtrip(tmp_path: Path) -> None:
    """Encrypt a file and decrypt it back verifying byte identity."""
    dk, ek = get_or_create_kem_keys()
    src = tmp_path / "raw.csv"
    src.write_bytes(b"col1,col2\n1.0,2.0\n3.0,4.0")
    enc = tmp_path / "raw.csv.enc"
    dec = tmp_path / "decrypted.csv"

    encrypt_file(src, enc, kem_ek=ek)
    assert enc.exists()
    assert enc.read_bytes() != src.read_bytes()

    decrypt_file(enc, dec, kem_dk=dk)
    assert dec.read_bytes() == src.read_bytes()


def test_aes_gcm_roundtrip() -> None:
    key = os.urandom(32)
    plaintext = b"neurodevelopmental benchmark results"
    nonce, ct = encrypt_blob(plaintext, key)
    recovered = decrypt_blob(nonce, ct, key)
    assert recovered == plaintext


def test_kyber_encapsulation() -> None:
    ss, ct = encapsulate()
    assert len(ss) == 32
    assert len(ct) > 0
