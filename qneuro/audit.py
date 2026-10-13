"""Append-only audit chain with cryptographic hash linking and ML-DSA-65 signatures."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from qneuro.vault import get_or_create_dsa_keys, sign_data, verify_signature

_GENESIS_HASH = "0" * 64


class AuditRecord(BaseModel):
    """Cryptographically signed, hash-chained log entry."""
    index: int
    timestamp: datetime.datetime
    actor: str = "agent"
    action: str
    payload_hash: str
    prev_hash: str
    entry_hash: str
    signature_hex: str = ""
    public_key_hex: str = ""
    # Optional metadata fields for traceability
    run_id: str | None = None
    step_number: int | None = None
    detail: str = ""
    reason: str | None = None

    @property
    def sequence(self) -> int:
        """Alias for backwards compatibility."""
        return self.index


def compute_payload_hash(payload: Any) -> str:
    """Compute deterministic SHA-256 hash of payload."""
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_entry_hash(entry_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 of the entry fields (excluding entry_hash and signature)."""
    fields_to_hash = {
        "index": entry_dict["index"],
        "timestamp": entry_dict["timestamp"],
        "actor": entry_dict.get("actor", "agent"),
        "action": entry_dict["action"],
        "payload_hash": entry_dict["payload_hash"],
        "prev_hash": entry_dict["prev_hash"],
        "run_id": entry_dict.get("run_id"),
        "step_number": entry_dict.get("step_number"),
        "detail": entry_dict.get("detail", ""),
        "reason": entry_dict.get("reason"),
    }
    canonical = json.dumps(fields_to_hash, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class AuditChain:
    """Append-only tamper-evident audit ledger backed by a JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._entries: list[AuditRecord] = []
        if self._path.parent:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        self._entries.clear()
        content = self._path.read_text(encoding="utf-8").strip()
        if not content:
            return
        for line in content.splitlines():
            line_str = line.strip()
            if line_str:
                raw_dict = json.loads(line_str)
                # Normalize index vs sequence
                if "index" not in raw_dict and "sequence" in raw_dict:
                    raw_dict["index"] = raw_dict["sequence"]
                if "payload_hash" not in raw_dict:
                    raw_dict["payload_hash"] = compute_payload_hash(raw_dict.get("detail", ""))
                self._entries.append(AuditRecord.model_validate(raw_dict))

    @property
    def last_hash(self) -> str:
        if not self._entries:
            return _GENESIS_HASH
        return self._entries[-1].entry_hash

    @property
    def length(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[AuditRecord]:
        return list(self._entries)

    def append(
        self,
        action: str,
        actor: str = "agent",
        payload: Any = None,
        run_id: str | None = None,
        detail: str = "",
        step_number: int | None = None,
        reason: str | None = None,
    ) -> AuditRecord:
        """Create, hash, sign, and append a new audit record."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        idx = len(self._entries)
        now = datetime.datetime.now(datetime.timezone.utc)
        payload_data = payload if payload is not None else detail
        p_hash = compute_payload_hash(payload_data)

        entry_data = {
            "index": idx,
            "timestamp": now.isoformat(),
            "actor": actor,
            "action": action,
            "payload_hash": p_hash,
            "prev_hash": self.last_hash,
            "run_id": run_id,
            "step_number": step_number,
            "detail": detail,
            "reason": reason,
        }
        e_hash = compute_entry_hash(entry_data)
        entry_data["entry_hash"] = e_hash

        # Sign entry hash with ML-DSA-65
        sk, pk = get_or_create_dsa_keys()
        sig_hex, pk_hex = sign_data(e_hash.encode("utf-8"), sk=sk, pk=pk)
        entry_data["signature_hex"] = sig_hex
        entry_data["public_key_hex"] = pk_hex

        record = AuditRecord.model_validate(entry_data)
        self._entries.append(record)

        with self._path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        return record

    def verify_chain(self) -> tuple[bool, int | None, str | None]:
        """Walk the chain and verify hash continuity, record digests, and signatures.

        Returns:
            (is_valid, failure_index, failure_reason)
        """
        prev_hash = _GENESIS_HASH
        for idx, entry in enumerate(self._entries):
            # Check sequential index
            if entry.index != idx:
                return False, idx, f"Index discontinuity: expected {idx}, found {entry.index}"

            # Check previous hash link
            if entry.prev_hash != prev_hash:
                return (
                    False,
                    idx,
                    f"Broken hash link at index {idx}: prev_hash ({entry.prev_hash[:12]}...) "
                    f"does not match expected previous entry ({prev_hash[:12]}...)",
                )

            # Recompute entry hash
            if entry.signature_hex:
                entry_dict = {
                    "index": entry.index,
                    "timestamp": entry.timestamp.isoformat(),
                    "actor": entry.actor,
                    "action": entry.action,
                    "payload_hash": entry.payload_hash,
                    "prev_hash": entry.prev_hash,
                    "run_id": entry.run_id,
                    "step_number": entry.step_number,
                    "detail": entry.detail,
                    "reason": entry.reason,
                }
                recomputed = compute_entry_hash(entry_dict)
            else:
                legacy_dict = {
                    "sequence": entry.index,
                    "timestamp": entry.timestamp.isoformat(),
                    "action": entry.action,
                    "run_id": entry.run_id,
                    "detail": entry.detail,
                    "step_number": entry.step_number,
                    "reason": entry.reason,
                    "prev_hash": entry.prev_hash,
                }
                recomputed = hashlib.sha256(json.dumps(legacy_dict, sort_keys=True, default=str).encode()).hexdigest()

            if recomputed != entry.entry_hash:
                return (
                    False,
                    idx,
                    f"Tampered record at index {idx} ({entry.action}): payload fields modified",
                )

            # Verify ML-DSA signature if present
            if entry.signature_hex and entry.public_key_hex:
                valid_sig = verify_signature(
                    entry.entry_hash.encode("utf-8"),
                    entry.signature_hex,
                    entry.public_key_hex,
                )
                if not valid_sig:
                    return (
                        False,
                        idx,
                        f"Cryptographic signature verification failed at index {idx} ({entry.action})",
                    )

            prev_hash = entry.entry_hash

        return True, None, None

    def verify(self) -> bool:
        """Boolean status of chain integrity."""
        valid, _, _ = self.verify_chain()
        return valid

    def verify_detailed(self) -> tuple[bool, str | None, int | None]:
        """Legacy compatibility method returning (valid, error_msg, failed_index)."""
        valid, idx, reason = self.verify_chain()
        return valid, reason, idx


def verify_chain(path: Path | str) -> tuple[bool, int | None, str | None]:
    """Standalone chain verification function."""
    ac = AuditChain(path)
    return ac.verify_chain()
