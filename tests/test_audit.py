"""Tests for the append-only cryptographic audit chain."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from qneuro.audit import AuditChain, verify_chain


@pytest.fixture()
def chain_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.jsonl"


def test_empty_chain_verifies(chain_path: Path) -> None:
    chain = AuditChain(chain_path)
    assert chain.verify()
    assert chain.length == 0
    valid, idx, reason = chain.verify_chain()
    assert valid is True
    assert idx is None
    assert reason is None


def test_append_and_verify(chain_path: Path) -> None:
    chain = AuditChain(chain_path)
    chain.append(action="step_1", detail="gateway initialized", step_number=1, reason="Explore data")
    chain.append(action="step_2", run_id="run_42", detail="kernel computed", step_number=2, reason="Fidelity matrix")
    chain.append(action="step_3", run_id="run_42", detail="guard executed", step_number=3, reason="Arbitrate")
    assert chain.length == 3
    assert chain.verify()
    valid, idx, reason = chain.verify_chain()
    assert valid is True
    assert idx is None
    assert reason is None


def test_reload_preserves_chain(chain_path: Path) -> None:
    chain = AuditChain(chain_path)
    chain.append(action="a")
    chain.append(action="b")
    reloaded = AuditChain(chain_path)
    assert reloaded.length == 2
    assert reloaded.verify()


def test_editing_middle_entry_fails_at_exact_index(chain_path: Path) -> None:
    """Modifying entry at index 1 in a 3-entry chain fails verification at index 1."""
    chain = AuditChain(chain_path)
    chain.append(action="step_0", detail="initialization")
    chain.append(action="step_1", detail="quantum_kernel_run")
    chain.append(action="step_2", detail="classical_benchmark")

    # Read lines and tamper only the middle entry (index 1)
    lines = chain_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    middle_dict = json.loads(lines[1])
    middle_dict["detail"] = "tampered_quantum_kernel_run"
    lines[1] = json.dumps(middle_dict)
    chain_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify fails at index 1
    valid, fail_idx, fail_reason = verify_chain(chain_path)
    assert valid is False
    assert fail_idx == 1
    assert "modified" in (fail_reason or "").lower() or "tampered" in (fail_reason or "").lower() or "signature" in (fail_reason or "").lower()


def test_deleting_entry_fails_verification(chain_path: Path) -> None:
    """Removing an entry from the chain creates a discontinuity that fails verification."""
    chain = AuditChain(chain_path)
    chain.append(action="first")
    chain.append(action="second")
    chain.append(action="third")

    # Delete the second line
    lines = chain_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    tampered_lines = [lines[0], lines[2]]
    chain_path.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")

    valid, fail_idx, fail_reason = verify_chain(chain_path)
    assert valid is False
    # The new second entry (previously third) has index 2 (expected 1) or prev_hash mismatch
    assert fail_idx == 1
