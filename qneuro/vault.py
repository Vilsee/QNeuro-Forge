"""Post-quantum cryptography module — ML-DSA-65 signatures, ML-KEM-768 encapsulation, and AES-256-GCM."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from dilithium_py.ml_dsa import ML_DSA_65
from kyber_py.ml_kem import ML_KEM_768

from qneuro.models.schemas import BenchmarkRun, SignedRun
from qneuro.types import SignedReport

KEYS_DIR = Path(".qneuro") / "keys"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Type {type(obj)} not serializable")


def canonical_json(data: Any) -> bytes:
    """Serialize object into canonical deterministic JSON (sorted keys, compact separators)."""
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _ensure_keys_dir() -> Path:
    """Ensure .qneuro/keys directory exists with restrictive permissions."""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(KEYS_DIR, 0o700)
    except Exception:
        pass
    return KEYS_DIR


def get_or_create_dsa_keys() -> tuple[bytes, bytes]:
    """Load or generate ML-DSA-65 signing keys.

    Priority:
    1. Environment variables QNEURO_SIGN_SK and QNEURO_SIGN_PK (hex).
    2. Files in .qneuro/keys/ml_dsa_sk.bin and ml_dsa_pk.bin (0600 permissions).
    3. Generate new keypair and persist to .qneuro/keys.
    """
    sk_hex = os.environ.get("QNEURO_SIGN_SK")
    pk_hex = os.environ.get("QNEURO_SIGN_PK")
    if sk_hex and pk_hex:
        return bytes.fromhex(sk_hex), bytes.fromhex(pk_hex)

    keys_dir = _ensure_keys_dir()
    sk_file = keys_dir / "ml_dsa_sk.bin"
    pk_file = keys_dir / "ml_dsa_pk.bin"

    if sk_file.exists() and pk_file.exists():
        return sk_file.read_bytes(), pk_file.read_bytes()

    pk, sk = ML_DSA_65.keygen()
    sk_file.write_bytes(sk)
    pk_file.write_bytes(pk)
    try:
        os.chmod(sk_file, 0o600)
        os.chmod(pk_file, 0o600)
    except Exception:
        pass

    return sk, pk


def get_or_create_kem_keys() -> tuple[bytes, bytes]:
    """Load or generate ML-KEM-768 key encapsulation keys.

    Priority:
    1. Environment variables QNEURO_KEM_DK and QNEURO_KEM_EK (hex).
    2. Files in .qneuro/keys/ml_kem_dk.bin and ml_kem_ek.bin (0600 permissions).
    3. Generate new keypair and persist to .qneuro/keys.
    """
    dk_hex = os.environ.get("QNEURO_KEM_DK")
    ek_hex = os.environ.get("QNEURO_KEM_EK")
    if dk_hex and ek_hex:
        return bytes.fromhex(dk_hex), bytes.fromhex(ek_hex)

    keys_dir = _ensure_keys_dir()
    dk_file = keys_dir / "ml_kem_dk.bin"
    ek_file = keys_dir / "ml_kem_ek.bin"

    if dk_file.exists() and ek_file.exists():
        return dk_file.read_bytes(), ek_file.read_bytes()

    ek, dk = ML_KEM_768.keygen()
    dk_file.write_bytes(dk)
    ek_file.write_bytes(ek)
    try:
        os.chmod(dk_file, 0o600)
        os.chmod(ek_file, 0o600)
    except Exception:
        pass

    return dk, ek


def derive_aes_key(
    shared_secret: bytes,
    salt: bytes | None = None,
    info: bytes = b"qneuro-aes-256-gcm-key",
) -> bytes:
    """Derive 256-bit symmetric encryption key from ML-KEM shared secret using HKDF-SHA256."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=info,
    )
    return hkdf.derive(shared_secret)


def encrypt_payload(
    plaintext: bytes,
    kem_ek: bytes | None = None,
) -> bytes:
    """Encrypt payload using ML-KEM-768 shared secret derived to AES-256-GCM.

    Format: [1088 bytes KEM ciphertext] + [12 bytes AES-GCM nonce] + [ciphertext + 16 bytes tag]
    """
    if kem_ek is None:
        _, kem_ek = get_or_create_kem_keys()

    shared_secret, kem_ciphertext = ML_KEM_768.encaps(kem_ek)
    aes_key = derive_aes_key(shared_secret)

    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    return kem_ciphertext + nonce + ciphertext


def decrypt_payload(
    encrypted_blob: bytes,
    kem_dk: bytes | None = None,
) -> bytes:
    """Decrypt payload using ML-KEM-768 decapsulation and AES-256-GCM."""
    if kem_dk is None:
        kem_dk, _ = get_or_create_kem_keys()

    if len(encrypted_blob) < 1088 + 12 + 16:
        raise ValueError("Encrypted blob is too short to contain valid headers and ciphertext")

    kem_ciphertext = encrypted_blob[:1088]
    nonce = encrypted_blob[1088 : 1088 + 12]
    ciphertext = encrypted_blob[1088 + 12 :]

    shared_secret = ML_KEM_768.decaps(kem_dk, kem_ciphertext)
    aes_key = derive_aes_key(shared_secret)

    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ciphertext, None)


def sign_data(
    data: bytes | Any,
    sk: bytes | None = None,
    pk: bytes | None = None,
) -> tuple[str, str]:
    """Sign canonical serialization using ML-DSA-65. Returns (signature_hex, public_key_hex)."""
    if sk is None or pk is None:
        sk_auto, pk_auto = get_or_create_dsa_keys()
        sk = sk or sk_auto
        pk = pk or pk_auto

    payload = canonical_json(data) if not isinstance(data, (bytes, bytearray)) else bytes(data)
    sig = ML_DSA_65.sign(sk, payload)
    return sig.hex(), pk.hex()


def verify_signature(
    data: bytes | Any,
    signature_hex: str,
    public_key_hex: str,
) -> bool:
    """Verify ML-DSA-65 signature over canonical serialization."""
    try:
        payload = canonical_json(data) if not isinstance(data, (bytes, bytearray)) else bytes(data)
        return ML_DSA_65.verify(
            bytes.fromhex(public_key_hex),
            payload,
            bytes.fromhex(signature_hex),
        )
    except Exception:
        return False


def get_public_key_fingerprint(public_key_hex: str) -> str:
    """Generate human-readable SHA-256 fingerprint for public key."""
    raw = bytes.fromhex(public_key_hex)
    digest = hashlib.sha256(raw).hexdigest()
    chunks = [digest[i : i + 4] for i in range(0, len(digest), 4)]
    return "SHA256:" + ":".join(chunks[:8])


def encrypt_file(src_path: Path | str, dst_path: Path | str, kem_ek: bytes | None = None) -> None:
    """Encrypt a dataset or report file to destination."""
    src = Path(src_path)
    dst = Path(dst_path)
    plaintext = src.read_bytes()
    encrypted = encrypt_payload(plaintext, kem_ek)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(encrypted)


def decrypt_file(src_path: Path | str, dst_path: Path | str, kem_dk: bytes | None = None) -> None:
    """Decrypt an encrypted dataset or report file to destination."""
    src = Path(src_path)
    dst = Path(dst_path)
    ciphertext = src.read_bytes()
    plaintext = decrypt_payload(ciphertext, kem_dk)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(plaintext)


def _load_signing_keys() -> tuple[bytes, bytes]:
    return get_or_create_dsa_keys()


def sign_run(run: BenchmarkRun) -> SignedRun:
    """Sign canonical JSON of BenchmarkRun with ML-DSA-65."""
    sig_hex, pk_hex = sign_data(run)
    return SignedRun(
        run=run,
        signature_hex=sig_hex,
        public_key_hex=pk_hex,
        algorithm="ML-DSA-65",
    )


def verify_run(signed: SignedRun) -> bool:
    """Verify ML-DSA-65 signature on signed run."""
    return verify_signature(signed.run, signed.signature_hex, signed.public_key_hex)


def encapsulate() -> tuple[bytes, bytes]:
    """Generate shared secret via ML-KEM-768."""
    _, ek = get_or_create_kem_keys()
    ss, ct = ML_KEM_768.encaps(ek)
    return ss, ct


def encrypt_blob(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Direct AES-256-GCM encryption with provided 32-byte key."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, None)
    return nonce, ct


def decrypt_blob(nonce: bytes, ciphertext: bytes, key: bytes) -> bytes:
    """Direct AES-256-GCM decryption."""
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)
