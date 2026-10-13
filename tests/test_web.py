"""Tests for FastAPI dashboard, states, and REST API endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qneuro.web.app import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_dashboard_renders_with_real_run(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "Instrument Ledger" in text
    assert "AUROC" in text
    assert "plot-svg" in text
    assert "SIGNATURE VALID" in text
    assert "ML-DSA-65" in text


def test_empty_state(client: TestClient) -> None:
    response = client.get("/?state=empty")
    assert response.status_code == 200
    text = response.text
    assert "No benchmark runs recorded in ledger" in text
    assert "python -m qneuro.agent" in text


def test_running_state(client: TestClient) -> None:
    response = client.get("/?state=running")
    assert response.status_code == 200
    text = response.text
    assert "step 3 of 8" in text


def test_error_state(client: TestClient) -> None:
    response = client.get("/?state=error")
    assert response.status_code == 200
    text = response.text
    assert "Benchmark failure:" in text
    assert "Retry command:" in text


def test_tampered_state(client: TestClient) -> None:
    response = client.get("/?state=tampered")
    assert response.status_code == 200
    text = response.text
    assert "TAMPER DETECTED:" in text


def test_verify_file_endpoint(client: TestClient) -> None:
    # Pick a valid signed run artifact
    runs = [p for p in Path("runs").glob("**/*.json") if "audit" not in p.name and "temp" not in p.name]
    valid_runs = []
    for r in runs:
        try:
            d = json.loads(r.read_text(encoding="utf-8"))
            if ("run" in d and "signature_hex" in d) or ("report_id" in d and "signature_hex" in d):
                valid_runs.append(r)
        except Exception:
            pass
    assert len(valid_runs) > 0
    latest_run = valid_runs[-1]
    raw = latest_run.read_bytes()

    # Valid upload
    res_valid = client.post(
        "/verify",
        files={"run_file": (latest_run.name, raw, "application/json")},
    )
    assert res_valid.status_code == 200
    assert "SIGNATURE VALID" in res_valid.text

    # Tampered upload (flip a byte)
    tampered_raw = bytearray(raw)
    tampered_raw[30] ^= 0x01
    res_tampered = client.post(
        "/verify",
        files={"run_file": ("tampered.json", bytes(tampered_raw), "application/json")},
    )
    assert res_tampered.status_code == 200
    assert "TAMPER DETECTED:" in res_tampered.text or "TAMPERED" in res_tampered.text


def test_api_runs_endpoint(client: TestClient) -> None:
    """Test GET /api/runs returns JSON list of runs."""
    res = client.get("/api/runs")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first_run = data[0]
    assert "id" in first_run
    assert "timestamp" in first_run
    assert "verified" in first_run


def test_api_run_details_and_audit(client: TestClient) -> None:
    """Test GET /api/runs/{id} and /api/runs/{id}/audit."""
    list_res = client.get("/api/runs")
    run_id = list_res.json()[0]["id"]

    details_res = client.get(f"/api/runs/{run_id}")
    assert details_res.status_code == 200
    details = details_res.json()
    assert details["run_id"] == run_id

    audit_res = client.get(f"/api/runs/{run_id}/audit")
    assert audit_res.status_code == 200
    audit_entries = audit_res.json()
    assert isinstance(audit_entries, list)


def test_api_verify_post(client: TestClient) -> None:
    """Test POST /api/verify with upload."""
    runs = [p for p in Path("runs").glob("**/*.json") if "audit" not in p.name and "temp" not in p.name]
    valid_runs = []
    for r in runs:
        try:
            d = json.loads(r.read_text(encoding="utf-8"))
            if ("run" in d and "signature_hex" in d) or ("report_id" in d and "signature_hex" in d):
                valid_runs.append(r)
        except Exception:
            pass
    assert len(valid_runs) > 0
    latest_run = valid_runs[-1]
    raw = latest_run.read_bytes()

    res = client.post(
        "/api/verify",
        files={"run_file": (latest_run.name, raw, "application/json")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["signature_valid"] is True
    assert data["algorithm"] == "ML-DSA-65"
    assert data["public_key_fingerprint"].startswith("SHA256:")
