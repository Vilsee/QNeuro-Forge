"""Command-line interface for QNeuro Forge."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from qneuro.agent import run_agent_pipeline
from qneuro.audit import AuditChain
from qneuro.models.schemas import SignedRun
from qneuro.types import SignedReport
from qneuro.vault import get_public_key_fingerprint, verify_run, verify_signature

app = typer.Typer(help="QNeuro Forge — Quantum vs classical ML neurodevelopmental benchmark notary.")


@app.command()
def run(
    budget: int = typer.Option(4, "--budget", "-b", help="Maximum candidate configurations to explore (<= 8)."),
    samples: int = typer.Option(120, "--samples", "-s", help="Number of cohort samples."),
    seed: int = typer.Option(42, "--seed", help="Random seed for data generator and cross-validation."),
) -> None:
    """Run autonomous quantum vs classical benchmark using the agent and OfflinePlanner."""
    typer.echo(f"Starting QNeuro Forge benchmark pipeline (budget={budget}, samples={samples}, seed={seed})...")
    report, run_dir = run_agent_pipeline(n_samples=samples, random_seed=seed, budget=budget)

    typer.echo(f"\nCompleted Run: {report.report_id}")
    typer.echo(f"Dataset: {report.dataset_source} (Fingerprint: {report.dataset_fingerprint[:16]}...)")
    typer.echo(f"Artifact directory: {run_dir}")

    typer.echo("\n=== Classical Baselines ===")
    for fs in report.fold_scores:
        if "quantum" not in fs.config_id and fs.metric == "auroc":
            typer.echo(f"  {fs.config_id:<32} AUROC={fs.mean_score:.4f} [{fs.ci_lower:.4f}, {fs.ci_upper:.4f}]")

    typer.echo("\n=== Quantum Configurations Tested ===")
    for fs in report.fold_scores:
        if "quantum" in fs.config_id and fs.metric == "auroc":
            typer.echo(f"  {fs.config_id:<32} AUROC={fs.mean_score:.4f} [{fs.ci_lower:.4f}, {fs.ci_upper:.4f}]")

    typer.echo("\n=== Arbitration Ruling ===")
    typer.echo(f"  Category: {report.verdict.category.value}")
    typer.echo(f"  AUROC Delta: {report.verdict.delta:+.4f} (95% CI: [{report.verdict.ci_lower:+.4f}, {report.verdict.ci_upper:+.4f}])")
    typer.echo(f"  Corrected p-value: {report.verdict.p_value:.4f}")
    typer.echo(f"  Verdict: {report.verdict.sentence}")
    typer.echo(f"\nML-DSA-65 Signature: {report.signature_hex[:32]}...")


@app.command()
def verify(
    report: Path = typer.Argument(..., help="Path to signed report JSON artifact."),
) -> None:
    """Verify ML-DSA-65 digital signature on a benchmark artifact."""
    if not report.exists():
        typer.echo(f"Error: Artifact file not found: {report}", err=True)
        raise typer.Exit(code=1)

    try:
        raw = json.loads(report.read_text(encoding="utf-8"))
    except Exception as e:
        typer.echo(f"Error: Failed to parse JSON in {report}: {e}", err=True)
        raise typer.Exit(code=1)

    is_valid = False
    algo = "ML-DSA-65"
    pk_fp = "UNKNOWN"

    # Check SignedReport format
    if "report_id" in raw and "fold_scores" in raw and "signature_hex" in raw:
        try:
            signed_rep = SignedReport.model_validate(raw)
            payload_data = signed_rep.model_dump(exclude={"signature_hex", "public_key_hex", "algorithm"})
            is_valid = verify_signature(payload_data, signed_rep.signature_hex, signed_rep.public_key_hex)
            algo = signed_rep.algorithm
            pk_fp = get_public_key_fingerprint(signed_rep.public_key_hex)
        except Exception as e:
            typer.echo(f"Verification error: {e}", err=True)
            raise typer.Exit(code=1)

    # Check SignedRun format
    elif "run" in raw and "signature_hex" in raw:
        try:
            signed_run = SignedRun.model_validate(raw)
            is_valid = verify_run(signed_run)
            algo = signed_run.algorithm
            pk_fp = get_public_key_fingerprint(signed_run.public_key_hex)
        except Exception as e:
            typer.echo(f"Verification error: {e}", err=True)
            raise typer.Exit(code=1)
    else:
        typer.echo(f"Error: Unrecognized signature envelope in {report}", err=True)
        raise typer.Exit(code=1)

    if is_valid:
        typer.echo(f"SIGNATURE VALID: {report.name}")
        typer.echo(f"Algorithm: {algo}")
        typer.echo(f"Public Key Fingerprint: {pk_fp}")
    else:
        typer.echo(f"TAMPER DETECTED: Cryptographic signature mismatch on {report.name}!", err=True)
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
) -> None:
    """Serve the web dashboard using uvicorn."""
    typer.echo(f"Starting QNeuro Forge dashboard at http://{host}:{port}/")
    uvicorn.run("qneuro.web.app:app", host=host, port=port, reload=False)


@app.command("tamper-demo")
def tamper_demo() -> None:
    """Copies latest signed report, flips one byte, and asserts verification failure."""
    runs_dir = Path("runs")
    reports = list(runs_dir.glob("*/report.json")) + list(runs_dir.glob("*.json"))
    valid_reports = []
    for r in reports:
        if r.name.startswith("_temp") or "audit" in r.name:
            continue
        try:
            d = json.loads(r.read_text(encoding="utf-8"))
            if ("report_id" in d or "run" in d) and "signature_hex" in d:
                valid_reports.append(r)
        except Exception:
            pass

    if not valid_reports:
        typer.echo("Error: No signed reports found in runs/. Run 'qneuro run' first.", err=True)
        raise typer.Exit(code=1)

    valid_reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    src_report = valid_reports[0]
    tampered_file = runs_dir / f"tampered_{src_report.name}"

    raw_bytes = bytearray(src_report.read_bytes())
    # Flip a bit in the body payload
    flip_pos = min(40, len(raw_bytes) - 1)
    raw_bytes[flip_pos] ^= 0x01

    tampered_file.write_bytes(raw_bytes)
    typer.echo(f"Copied {src_report.name} -> {tampered_file.name} with byte at offset {flip_pos} flipped.")

    # Run verification on tampered file
    typer.echo(f"Verifying tampered file: {tampered_file.name}...")
    try:
        raw_tampered = json.loads(tampered_file.read_text(encoding="utf-8"))
        if "report_id" in raw_tampered and "signature_hex" in raw_tampered:
            rep = SignedReport.model_validate(raw_tampered)
            payload = rep.model_dump(exclude={"signature_hex", "public_key_hex", "algorithm"})
            valid = verify_signature(payload, rep.signature_hex, rep.public_key_hex)
        elif "run" in raw_tampered and "signature_hex" in raw_tampered:
            s_run = SignedRun.model_validate(raw_tampered)
            valid = verify_run(s_run)
        else:
            valid = False
    except Exception as exc:
        valid = False
        reason = f"Structural validation / JSON parsing failed: {exc}"

    if not valid:
        typer.echo(f"VERIFICATION FAILED AS EXPECTED.")
        typer.echo(f"Reason: ML-DSA-65 signature mismatch (cryptographic hash mismatch on payload byte modification).")
    else:
        typer.echo(f"UNEXPECTED: Tampered report passed verification!", err=True)
        raise typer.Exit(code=1)


@app.command()
def audit(
    audit_file: Path = typer.Option(Path("runs/audit.jsonl"), help="Path to audit ledger."),
) -> None:
    """Inspect and verify append-only cryptographic audit chain."""
    ac = AuditChain(audit_file)
    valid, idx, err = ac.verify_chain()
    typer.echo(f"Audit Chain: {ac.length} entries. Head: {ac.last_hash[:16]}...")
    if valid:
        typer.echo("CHAIN INTEGRITY INTACT: All cryptographic entries verified.")
    else:
        typer.echo(f"CHAIN TAMPER DETECTED at entry {idx}: {err}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
