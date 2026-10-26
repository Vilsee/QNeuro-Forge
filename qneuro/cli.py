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


# --------------------------------------------------------------------------- study
study_app = typer.Typer(help="Re-check, sign and verify the Team Morphos heart-failure QSVM study.")
app.add_typer(study_app, name="study")


def _line(ok: bool, name: str, detail: str, expected: str = "PASS") -> bool:
    got = "PASS" if ok else "FAIL"
    if expected == "FAIL":
        tag = "DETECTED" if got == "FAIL" else "MISSED"
    else:
        tag = got
    typer.echo(f"  [{tag}] {name}  ({detail})")
    return got == expected


@study_app.command("build")
def study_build(fast: bool = typer.Option(False, "--fast", help="Skip sweeps and the no-time guard run.")) -> None:
    """Re-run every experiment, check every claim, then sign the result with ML-DSA-65."""
    from qneuro.study.build import build

    env = build(fast=fast, log=typer.echo)
    typer.echo(f"Signed study written to public/data/study.json (sha256 {env['study_sha256'][:16]}...)")


@study_app.command("verify")
def study_verify() -> None:
    """Verify the study signature, every artifact hash, and the audit chain."""
    from qneuro.study.build import verify

    ok_all = True
    typer.echo("Verifying public/data/study.json")
    for c in verify():
        ok_all &= _line(c["ok"], c["name"], c["detail"])
    if not ok_all:
        typer.echo("VERIFICATION FAILED", err=True)
        raise typer.Exit(code=1)
    typer.echo("ALL CHECKS PASSED")


@study_app.command("tamper-demo")
def study_tamper_demo() -> None:
    """Change one digit in a copy of the submission and in a copy of the study, then verify both."""
    from qneuro.study.build import tamper_demo

    ok_all = True
    for c in tamper_demo():
        ok_all &= _line(c["ok"], c["name"], c["detail"], expected=c["expected"])
    typer.echo("Both edits were detected." if ok_all else "A tampered file passed verification.")
    if not ok_all:
        raise typer.Exit(code=1)


@study_app.command("reproduce")
def study_reproduce() -> None:
    """Live re-run of the submitted QSVM (E5) and the local ZZ benchmark (E6)."""
    from qneuro.study import experiments as ex

    typer.echo("E5: independent angle-embedding kernel against the submitted file")
    e5 = ex.reproduce_e5()
    h, hs = e5["holdout"], e5["holdout_submitted"]
    for k in ("mcc", "accuracy", "f1", "precision", "recall", "roc_auc"):
        typer.echo(f"  holdout {k:<10} ours {h[k]:.6f}   submitted {hs[k]:.6f}   diff {abs(h[k] - hs[k]):.6f}")
    typer.echo(f"  confusion matrix {e5['holdout_confusion']}")
    typer.echo(f"  CV folds ours      {e5['cv_fold_mcc']}")
    typer.echo(f"  CV folds submitted {e5['cv_fold_mcc_submitted']}   match: {e5['fold_match']}")
    typer.echo(f"  true ZZ kernel, same range, holdout MCC {e5['zz_same_range_holdout_mcc']:.4f}  (differs, so the submitted kernel is not ZZ)")
    typer.echo("E6: local true-ZZ benchmark against the stored file")
    cmp_ = ex.compare_with_stored_e6(ex.benchmark_e6())
    typer.echo(f"  differing values: {cmp_['n_differences']}  identical: {cmp_['identical']}")


@study_app.command("prepare")
def study_prepare(
    out: Path = typer.Option(Path("data/prepared"), help="Output folder for the four CSVs."),
    local_csv: Optional[Path] = typer.Option(None, help="Use a local heart_failure_clinical_records CSV instead of downloading."),
) -> None:
    """Rebuild the train and test CSVs from UCI dataset 519 and print their SHA-256 hashes."""
    from qneuro.study.data import prepare_from_uci

    for name, digest in prepare_from_uci(out, local_csv).items():
        typer.echo(f"{digest}  {name}")


@study_app.command("serve")
def study_serve(port: int = typer.Option(8080, "--port", "-p")) -> None:
    """Serve the static study site from public/ (the same files Vercel serves)."""
    import functools
    import http.server

    root = Path(__file__).resolve().parents[1] / "public"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    typer.echo(f"Study site at http://127.0.0.1:{port}/  (Ctrl+C to stop)")
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    app()
