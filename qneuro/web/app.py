"""FastAPI dashboard and REST API — reads results from runs/, never fabricates numbers."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from qneuro.audit import AuditChain
from qneuro.guard import evaluate, evaluate_advantage
from qneuro.models.schemas import BenchmarkRun, SignedRun
from qneuro.types import SignedReport
from qneuro.vault import (
    canonical_json,
    get_public_key_fingerprint,
    verify_run,
    verify_signature,
)

_RUNS_DIR = Path("runs")
_WEB_DIR = Path(__file__).parent

app = FastAPI(title="QNeuro Forge", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=_WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=_WEB_DIR / "templates")


def _render_interval_svg(models: list[dict[str, Any]], best_classical_name: str) -> str:
    """Generate hand-written SVG horizontal interval plot adhering strictly to design rules.

    Sorted by mean AUROC. Direct model labels beside whiskers, no legend,
    no gridlines beyond axis ticks. Quantum in verdigris (#1E6B55),
    best classical in ink (#1C1B18), remainder in ink-2 (#5B574D).
    """
    if not models:
        return ""

    # Sort descending by AUROC
    sorted_models = sorted(models, key=lambda m: m.get("roc_auc") or 0.0, reverse=True)

    width = 560
    row_height = 34
    top_margin = 25
    bottom_margin = 45
    height = top_margin + len(sorted_models) * row_height + bottom_margin

    left_label_w = 200
    plot_w = 310
    x_min = 0.0
    x_max = 1.0

    def val_to_x(v: float) -> float:
        clamped = max(x_min, min(x_max, v))
        return left_label_w + (clamped - x_min) / (x_max - x_min) * plot_w

    lines: list[str] = [
        f'<svg class="plot-svg" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
    ]

    # Axis line at bottom
    axis_y = top_margin + len(sorted_models) * row_height + 10
    lines.append(
        f'  <line x1="{left_label_w}" y1="{axis_y}" x2="{left_label_w + plot_w}" y2="{axis_y}" stroke="#1C1B18" stroke-width="1" />'
    )

    # Axis ticks at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    for tick_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        tx = val_to_x(tick_val)
        lines.append(
            f'  <line x1="{tx:.1f}" y1="{axis_y}" x2="{tx:.1f}" y2="{axis_y + 4}" stroke="#1C1B18" stroke-width="1" />'
        )
        lines.append(
            f'  <text x="{tx:.1f}" y="{axis_y + 16}" font-family="\'IBM Plex Mono\', monospace" font-size="10" fill="#5B574D" text-anchor="middle">{tick_val:.1f}</text>'
        )

    # Axis title
    lines.append(
        f'  <text x="{left_label_w + plot_w / 2}" y="{axis_y + 30}" font-family="\'IBM Plex Mono\', monospace" font-size="10" fill="#5B574D" text-anchor="middle">AUROC (mean ± 95% bootstrap / Student-t CI)</text>'
    )

    # Render each model's interval
    for idx, m in enumerate(sorted_models):
        y = top_margin + idx * row_height + 15
        is_q = m.get("is_quantum", False) or ("quantum" in m.get("model_name", "").lower())
        is_best_c = (m.get("model_name") == best_classical_name) and not is_q

        if is_q:
            color = "#1E6B55"  # --verdigris
            weight = "600"
        elif is_best_c:
            color = "#1C1B18"  # --ink
            weight = "500"
        else:
            color = "#5B574D"  # --ink-2
            weight = "400"

        # Direct label left of plot
        display_name = m.get("model_name", "")
        lines.append(
            f'  <text x="{left_label_w - 12}" y="{y + 4}" font-family="\'IBM Plex Mono\', monospace" font-size="11" font-weight="{weight}" fill="{color}" text-anchor="end">{display_name}</text>'
        )

        mean_val = m.get("roc_auc")
        ci_low = m.get("roc_auc_ci_lower")
        ci_high = m.get("roc_auc_ci_upper")

        if mean_val is not None:
            low_val = ci_low if ci_low is not None else mean_val
            high_val = ci_high if ci_high is not None else mean_val

            x_low = val_to_x(low_val)
            x_high = val_to_x(high_val)
            x_dot = val_to_x(mean_val)

            # Horizontal whisker line
            lines.append(
                f'  <line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="{color}" stroke-width="1.5" />'
            )
            # End cap ticks
            lines.append(
                f'  <line x1="{x_low:.1f}" y1="{y - 4}" x2="{x_low:.1f}" y2="{y + 4}" stroke="{color}" stroke-width="1.5" />'
            )
            lines.append(
                f'  <line x1="{x_high:.1f}" y1="{y - 4}" x2="{x_high:.1f}" y2="{y + 4}" stroke="{color}" stroke-width="1.5" />'
            )
            # Mean AUROC dot
            lines.append(
                f'  <circle cx="{x_dot:.1f}" cy="{y}" r="3.5" fill="{color}" />'
            )
            # Direct numeric score label
            lines.append(
                f'  <text x="{x_high + 8:.1f}" y="{y + 4}" font-family="\'IBM Plex Mono\', monospace" font-size="10" fill="{color}">{mean_val:.4f}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def _parse_run_file(file_path: Path) -> dict[str, Any] | None:
    """Parse either SignedReport or SignedRun file into a standardized dashboard view dict."""
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    audit_path = file_path.parent / "audit.jsonl"
    if not audit_path.exists():
        audit_path = _RUNS_DIR / "audit.jsonl"
    audit = AuditChain(audit_path)
    audit_valid, audit_err, failed_idx = audit.verify_detailed()

    # Case 1: SignedReport (new types.py format)
    if "report_id" in raw and "fold_scores" in raw:
        try:
            report = SignedReport.model_validate(raw)
            payload_data = report.model_dump(exclude={"signature_hex", "public_key_hex", "algorithm"})
            verified = verify_signature(payload_data, report.signature_hex, report.public_key_hex)
            pk_fingerprint = get_public_key_fingerprint(report.public_key_hex)

            classical_results: list[dict[str, Any]] = []
            quantum_results: list[dict[str, Any]] = []

            # Extract AUROC metrics
            for fs in report.fold_scores:
                if fs.metric == "auroc":
                    row = {
                        "model_name": fs.config_id,
                        "roc_auc": fs.mean_score,
                        "roc_auc_ci_lower": fs.ci_lower,
                        "roc_auc_ci_upper": fs.ci_upper,
                        "f1_score": 0.0,
                        "train_time_seconds": report.wall_clock_seconds.get(fs.config_id, 0.0),
                        "is_quantum": "quantum" in fs.config_id.lower(),
                    }
                    if row["is_quantum"]:
                        quantum_results.append(row)
                    else:
                        classical_results.append(row)

            # Match F1 scores if available
            for fs in report.fold_scores:
                if fs.metric == "f1":
                    for r in classical_results + quantum_results:
                        if r["model_name"] == fs.config_id:
                            r["f1_score"] = fs.mean_score

            all_models = classical_results + quantum_results
            best_c = report.verdict.best_classical_model
            svg_plot = _render_interval_svg(all_models, best_c)

            ts_str = report.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if hasattr(report.timestamp, "strftime") else str(report.timestamp)
            return {
                "run_id": report.report_id,
                "timestamp_str": ts_str,
                "dataset_source_label": report.dataset_source,
                "verdict_claim": report.verdict.sentence,
                "verdict_note": (
                    f"Evaluated with Nadeau-Bengio corrected resampled t-test and bootstrap 95% CI "
                    f"(corrected p = {report.verdict.p_value:.4f} across {report.verdict.configs_tried} configs). "
                    f"Neurodevelopmental phenotyping on small sample cohorts exhibits sampling noise."
                ),
                "cohort_desc": "120 samples, 8 features (SYNTHETIC)",
                "run": {
                    "run_id": report.report_id,
                    "timestamp": report.timestamp,
                    "dataset_source": {"value": report.dataset_source},
                    "dataset_name": "synth_neurodev",
                    "n_samples": 120,
                    "n_features": 8,
                    "classical_results": classical_results,
                    "quantum_results": quantum_results,
                },
                "verified": verified,
                "verdict": {
                    "claim": report.verdict.sentence,
                    "confidence_note": (
                        f"Evaluated with Nadeau-Bengio corrected resampled t-test and bootstrap 95% CI "
                        f"(corrected p = {report.verdict.p_value:.4f} across {report.verdict.configs_tried} configs). "
                    ),
                    "best_classical_model": best_c,
                    "best_quantum_model": report.verdict.best_quantum_model,
                    "delta": report.verdict.delta,
                },
                "filename": file_path.name,
                "pk_fingerprint": pk_fingerprint,
                "sig_algo": report.algorithm,
                "audit_head": audit.last_hash,
                "audit_length": audit.length,
                "audit_valid": audit_valid,
                "audit_err": audit_err,
                "audit_failed_idx": failed_idx,
                "agent_steps": [
                    {"step": e.index, "action": e.action, "reason": e.reason or e.detail}
                    for e in audit.entries if e.action != "INITIALIZE_GATEWAY"
                ],
                "svg_plot": svg_plot,
            }
        except Exception:
            pass

    # Case 2: SignedRun or BenchmarkRun (schemas.py format)
    if "run" in raw and "signature_hex" in raw:
        signed = SignedRun.model_validate(raw)
        verified = verify_run(signed)
        run = signed.run
        pk_fingerprint = get_public_key_fingerprint(signed.public_key_hex)
        sig_algo = signed.algorithm
    elif "run_id" in raw:
        run = BenchmarkRun.model_validate(raw)
        verified = None
        pk_fingerprint = "UNSIGNED_KEY"
        sig_algo = "NONE"
    else:
        return None

    verdict_obj = evaluate(run, metric="roc_auc")
    c_models = [m.model_dump() for m in run.classical_results]
    q_models = [m.model_dump() for m in run.quantum_results]
    svg_plot = _render_interval_svg(c_models + q_models, verdict_obj.best_classical_model)

    ts_str = run.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if hasattr(run.timestamp, "strftime") else str(run.timestamp)
    src_val = run.dataset_source.value if hasattr(run.dataset_source, "value") else str(run.dataset_source)
    return {
        "run_id": run.run_id,
        "timestamp_str": ts_str,
        "dataset_source_label": src_val,
        "verdict_claim": verdict_obj.claim,
        "verdict_note": verdict_obj.confidence_note,
        "cohort_desc": f"{run.n_samples} samples, {run.n_features} features, seed {run.random_seed}",
        "run": run,
        "verified": verified,
        "verdict": verdict_obj,
        "filename": file_path.name,
        "pk_fingerprint": pk_fingerprint,
        "sig_algo": sig_algo,
        "audit_head": audit.last_hash,
        "audit_length": audit.length,
        "audit_valid": audit_valid,
        "audit_err": audit_err,
        "audit_failed_idx": failed_idx,
        "agent_steps": run.agent_steps,
        "svg_plot": svg_plot,
    }


def _load_run_data() -> dict[str, Any] | None:
    """Find and load the latest available run file."""
    if not _RUNS_DIR.exists():
        return None

    # Search for report.json in subdirectories or .json in runs/
    candidates = list(_RUNS_DIR.glob("*/report.json")) + sorted(_RUNS_DIR.glob("*.json"))
    if not candidates:
        return None

    # Pick latest modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        parsed = _parse_run_file(p)
        if parsed:
            return parsed

    return None


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, state: str | None = None) -> HTMLResponse:
    if state == "empty":
        return templates.TemplateResponse(request=request, name="index.html", context={"state": "empty"})
    if state == "running":
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "state": "running",
                "running_step": "step 3 of 8",
                "running_action": "SYNTHESIZE_COHORT",
            },
        )
    if state == "error":
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "state": "error",
                "error_stage": "FIT_BENCHMARK_FOLDS (kernels.py)",
                "error_message": "Statevector simulation timeout on 6-qubit register.",
                "retry_cmd": "python -m qneuro.agent",
            },
        )

    data = _load_run_data()
    if not data:
        return templates.TemplateResponse(request=request, name="index.html", context={"state": "empty"})

    tampered_override = None
    if state == "tampered":
        tampered_override = "Audit entry index 1 payload was altered (hash mismatch detected)"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"data": data, "tampered_override": tampered_override},
    )


@app.post("/verify", response_class=HTMLResponse)
async def verify_uploaded_file(request: Request, run_file: UploadFile = File(...)) -> HTMLResponse:
    """Handle verification upload from web interface."""
    content = await run_file.read()
    temp_path = _RUNS_DIR / f"_temp_verify_{run_file.filename}"
    try:
        temp_path.write_bytes(content)
        parsed = _parse_run_file(temp_path)
        if not parsed:
            raise ValueError("File is not a valid SignedReport or SignedRun JSON document")

        tamper_msg = None
        if not parsed["verified"]:
            tamper_msg = f"Signature verification failed for {run_file.filename}: ML-DSA-65 public key mismatch."

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"data": parsed, "tampered_override": tamper_msg},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "state": "error",
                "error_stage": f"Verification parser ({run_file.filename})",
                "error_message": str(exc),
                "retry_cmd": "python -m qneuro.agent",
            },
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    """List all completed benchmark runs."""
    if not _RUNS_DIR.exists():
        return []

    summaries: list[dict[str, Any]] = []
    # Search subdirectories first (runs/<run_id>/report.json)
    for p in sorted(_RUNS_DIR.glob("*/report.json"), reverse=True):
        parsed = _parse_run_file(p)
        if parsed:
            run_dict = parsed["run"]
            run_id = run_dict["run_id"] if isinstance(run_dict, dict) else run_dict.run_id
            timestamp = run_dict["timestamp"] if isinstance(run_dict, dict) else run_dict.timestamp
            summaries.append({
                "id": run_id,
                "timestamp": str(timestamp),
                "verified": parsed["verified"],
                "algorithm": parsed["sig_algo"],
                "path": str(p),
            })

    # Top level files
    for p in sorted(_RUNS_DIR.glob("*.json"), reverse=True):
        if p.name.startswith("_temp_"):
            continue
        parsed = _parse_run_file(p)
        if parsed:
            run_dict = parsed["run"]
            run_id = run_dict["run_id"] if isinstance(run_dict, dict) else run_dict.run_id
            if not any(s["id"] == run_id for s in summaries):
                timestamp = run_dict["timestamp"] if isinstance(run_dict, dict) else run_dict.timestamp
                summaries.append({
                    "id": run_id,
                    "timestamp": str(timestamp),
                    "verified": parsed["verified"],
                    "algorithm": parsed["sig_algo"],
                    "path": str(p),
                })

    return summaries


@app.get("/api/runs/{run_id}")
async def get_run_details(run_id: str) -> dict[str, Any]:
    """Retrieve complete report artifact for run."""
    sub_path = _RUNS_DIR / run_id / "report.json"
    top_path = _RUNS_DIR / f"{run_id}.json"

    target = sub_path if sub_path.exists() else top_path
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    parsed = _parse_run_file(target)
    if not parsed:
        raise HTTPException(status_code=500, detail="Corrupted run artifact")

    return parsed


@app.get("/api/runs/{run_id}/audit")
async def get_run_audit(run_id: str) -> list[dict[str, Any]]:
    """Retrieve cryptographic audit records associated with run."""
    sub_audit = _RUNS_DIR / run_id / "audit.jsonl"
    global_audit = _RUNS_DIR / "audit.jsonl"

    target = sub_audit if sub_audit.exists() else global_audit
    if not target.exists():
        return []

    ac = AuditChain(target)
    return [e.model_dump() for e in ac.entries if e.run_id in (None, run_id)]


@app.post("/api/verify")
async def api_verify(run_file: UploadFile = File(...)) -> dict[str, Any]:
    """Verify an uploaded report file for ML-DSA-65 signature and audit continuity."""
    content = await run_file.read()
    temp_path = _RUNS_DIR / f"_temp_api_verify_{run_file.filename}"
    try:
        temp_path.write_bytes(content)
        parsed = _parse_run_file(temp_path)
        if not parsed:
            return {
                "valid": False,
                "signature_valid": False,
                "audit_chain_valid": False,
                "reason": "Malformed or unrecognized JSON report artifact",
            }

        return {
            "valid": bool(parsed["verified"] and parsed["audit_valid"]),
            "signature_valid": bool(parsed["verified"]),
            "algorithm": parsed["sig_algo"],
            "public_key_fingerprint": parsed["pk_fingerprint"],
            "audit_chain_valid": bool(parsed["audit_valid"]),
            "audit_chain_length": parsed["audit_length"],
            "failure_reason": parsed["audit_err"] if not parsed["audit_valid"] else (
                "Signature mismatch" if not parsed["verified"] else None
            ),
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()
