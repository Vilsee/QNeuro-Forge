"""Build, sign, verify and tamper-test the Morphos study ledger."""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from qneuro.audit import AuditChain, compute_payload_hash, verify_chain
from qneuro.study import experiments as ex
from qneuro.study.claims import STATUS_LABEL, build_claims
from qneuro.study.data import DATA, ROOT, load_split, sha256_file
from qneuro.vault import canonical_json, get_public_key_fingerprint, sign_data, verify_signature

PUBLIC = ROOT / "public" / "data"
RUNS = ROOT / "runs" / "morphos"

ROLES = {
    "original/hf_withtime_train.csv": "Training rows, with follow-up time (239 patients)",
    "original/hf_withtime_test.csv": "Holdout rows, with follow-up time (60 patients)",
    "original/hf_notime_train.csv": "Training rows, serum sodium instead of time",
    "original/hf_notime_test.csv": "Holdout rows, serum sodium instead of time",
    "scaled_0.3/hf_withtime_s03_train.csv": "Training rows, inputs shrunk to 0 to 0.3",
    "scaled_0.3/hf_withtime_s03_test.csv": "Holdout rows, inputs shrunk to 0 to 0.3",
    "scaled_0.3/hf_notime_s03_train.csv": "Training rows, no time, shrunk to 0 to 0.3",
    "scaled_0.3/hf_notime_s03_test.csv": "Holdout rows, no time, shrunk to 0 to 0.3",
    "bloq_outputs/E1_notime_config.json": "Raw Bloq platform output, experiment E1",
    "bloq_outputs/E2_withtime_config.json": "Raw Bloq platform output, experiment E2",
    "submissions/SUBMITTED_corrected.json": "The file submitted to the hackathon (E5)",
    "submissions/E5_original_export.json": "Bloq pipeline export before three text fields were corrected",
    "submissions/E6_local_zz_NOT_submitted.json": "Local ZZ benchmark, not submitted (E6)",
    "scripts_original/prepare_hf.py": "Data preparation script",
    "scripts_original/local_qsvm.py": "Local QSVM script",
    "scripts_original/sweep1_depth_scale.py": "Sweep 1 script (E3)",
    "scripts_original/sweep2_preprocessing.py": "Sweep 2 script (E4)",
    "scripts_original/zz_benchmark.py": "ZZ benchmark script (E6)",
}


def _plain(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    return o


def _diff(x: Any, y: Any, path: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(dict(field=f"{path}/{k}", before=x.get(k), after=y.get(k)))
            else:
                out += _diff(x[k], y[k], f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            out += _diff(u, v, f"{path}[{i}]")
    elif x != y:
        out.append(dict(field=path, before=x, after=y))
    return out


def manifest() -> list[dict[str, Any]]:
    rows = []
    for rel, role in ROLES.items():
        p = DATA / rel
        rows.append(dict(path=rel, sha256=sha256_file(p), bytes=p.stat().st_size, role=role))
    return rows


def _ledger(ctx: dict[str, Any], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {c["id"]: c for c in claims}
    e6p = ctx["e6"]["performance_summary"]["QSVM (Quantum Kernel)"]
    sub = json.loads((DATA / "submissions" / "SUBMITTED_corrected.json").read_text("utf-8"))
    sq = sub["performance_summary"]["QSVM (Quantum Kernel)"]
    e1 = json.loads((DATA / "bloq_outputs" / "E1_notime_config.json").read_text("utf-8"))["output"]
    e2 = json.loads((DATA / "bloq_outputs" / "E2_withtime_config.json").read_text("utf-8"))["output"]
    rows = [
        dict(id="E1", what="Bloq platform, QSVM with the default ZZ feature map, 4 qubits, depth 1", where="Bloq", data="no time (serum sodium)",
             cv_mcc=None, holdout_mcc=round(e1["mcc"], 4), eval_n=48, submitted="no", claim_ids=["C-E1", "C-GAP"]),
        dict(id="E2", what="Same platform settings, with follow-up time", where="Bloq", data="with time",
             cv_mcc=None, holdout_mcc=round(e2["mcc"], 4), eval_n=48, submitted="no", claim_ids=["C-E2", "C-GAP"]),
    ]
    if ctx.get("e3"):
        w = [r for r in ctx["e3"] if r["data"] == "withtime" and r["class_weight"] == "none"]
        rows.append(dict(id="E3", what="Local sweep over depth and input range, true ZZ kernel", where="Local", data="both",
                         cv_mcc=None, holdout_mcc=None, eval_n=239, submitted="no", claim_ids=["C-E3"],
                         span=f"CV MCC {min(r['cv_mcc'] for r in w):.2f} to {max(r['cv_mcc'] for r in w):.2f} (with time)"))
    if ctx.get("e4"):
        w = [r for r in ctx["e4"] if r["data"] == "withtime" and r["mode"] == "minmax"]
        b = max(w, key=lambda r: r["cv_mcc"])
        rows.append(dict(id="E4", what="Local sweep over preprocessing and range, repeated 5-fold CV", where="Local", data="both",
                         cv_mcc=b["cv_mcc"], holdout_mcc=b["test_mcc"], eval_n=239, submitted="no", claim_ids=["C-E4", "C-SELECT"],
                         span=f"best with-time range {b['scale']}"))
    rows.append(dict(id="E5", what="Bloq pipeline export: angle-embedding QSVM and five classical models", where="Bloq pipeline", data="with time",
                     cv_mcc=sq["cv_5fold"]["mcc_mean"], holdout_mcc=round(sq["holdout_test"]["mcc"], 4), eval_n=60, submitted="yes",
                     claim_ids=["C-E5-HOLD", "C-E5-CV", "C-E5-MAP", "C-E5-LR", "C-E5-SVM", "C-E5-TREES", "C-E5-PAIR", "C-PLATFORM"]))
    rows.append(dict(id="E6", what="Local true-ZZ QSVM and five classical models on identical folds", where="Local", data="with time",
                     cv_mcc=e6p["cv_5fold"]["mcc_mean"], holdout_mcc=round(e6p["holdout_test"]["mcc"], 4), eval_n=60, submitted="no",
                     claim_ids=["C-E6"]))
    for r in rows:
        sts = [by[c]["status"] for c in r["claim_ids"] if c in by and c != "C-PLATFORM"]
        good = {"reproduced", "supported"}
        if sts and all(x in good for x in sts):
            st = "reproduced"
        elif any(x in good for x in sts):
            st = "partly"
        else:
            st = "not_reproduced"
        r["status"], r["status_label"] = st, STATUS_LABEL[st]
        r["provenance"] = "Run on Bloq, not verifiable from outside" if r["where"].startswith("Bloq") else "Local, re-runnable here"
    return rows


def build(fast: bool = False, public_dir: Path | None = None, runs_dir: Path | None = None, log=print) -> dict[str, Any]:
    pub = public_dir or PUBLIC
    runs = runs_dir or RUNS
    runs.mkdir(parents=True, exist_ok=True)
    chain_path = runs / "audit.jsonl"
    if chain_path.exists():
        chain_path.unlink()
    chain = AuditChain(chain_path)
    step = [0]

    def note(action: str, payload: Any, detail: str, reason: str) -> None:
        step[0] += 1
        chain.append(action=action, actor="morphos-study", payload=payload, run_id="morphos", detail=detail,
                     step_number=step[0], reason=reason)

    log("1/8 hashing artifacts")
    man = manifest()
    for m in man:
        note("INGEST_ARTIFACT", dict(path=m["path"], sha256=m["sha256"]), f"{m['path']} sha256={m['sha256'][:16]}",
             "Fix the exact bytes of every input and output before any analysis runs.")
    art_dir = pub / "artifacts"
    if art_dir.exists():
        shutil.rmtree(art_dir)
    for m in man:
        dst = art_dir / m["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(DATA / m["path"], dst)

    ctx: dict[str, Any] = {}
    log("2/8 reproducing E5 (submitted QSVM)")
    ctx["e5"] = ex.reproduce_e5()
    note("REPRODUCE_E5", dict(holdout=ctx["e5"]["holdout"], folds=ctx["e5"]["cv_fold_mcc"]), "angle-embedding kernel re-implemented",
         "Check the submitted numbers with code that does not come from the platform.")
    log("3/8 reproducing E6 (local ZZ benchmark)")
    ctx["e6"] = ex.benchmark_e6()
    ctx["e6_compare"] = ex.compare_with_stored_e6(ctx["e6"])
    note("REPRODUCE_E6", ctx["e6_compare"], f"differences vs stored file: {ctx['e6_compare']['n_differences']}",
         "A fresh run must equal the stored file.")
    if not fast:
        log("4/8 sweeps E3 and E4")
        ctx["e3"] = ex.sweep_e3()
        ctx["e4"] = ex.sweep_e4()
        note("SWEEP_E3_E4", dict(e3=len(ctx["e3"]), e4=len(ctx["e4"])), "sweeps re-run",
             "Replace the report's approximate E3 and E4 tables with exact values.")
        log("5/8 leakage and platform-gap studies")
        ctx["leakage"] = ex.leakage_study()
        ctx["gap"] = ex.platform_gap_study()
        note("LEAKAGE_AND_GAP", dict(leak=ctx["leakage"]["paired"], gap=ctx["gap"]), "time-feature and platform-gap analyses",
             "Quantify what follow-up time contributes and test whether split luck explains E1 and E2.")
    ctx["sensitivity"] = ex.holdout_sensitivity(ctx["e5"]["holdout_confusion"])

    sub = json.loads((DATA / "submissions" / "SUBMITTED_corrected.json").read_text("utf-8"))
    orig = json.loads((DATA / "submissions" / "E5_original_export.json").read_text("utf-8"))
    d = _diff(orig, sub)
    ctx["integrity"] = dict(diff=d, only_text_fields_changed=all(isinstance(x["before"], str) and isinstance(x["after"], str) for x in d))
    note("DIFF_SUBMISSION", ctx["integrity"], f"{len(d)} fields differ", "Show exactly what was edited before submission.")

    log("6/8 running the Honesty Guard on repeated cross-validation")
    guard = {v: ex.guard_study(v) for v in (("withtime", "notime") if not fast else ("withtime",))}
    for v, g in guard.items():
        note(f"GUARD_VERDICT_{v.upper()}", g["verdicts"], g["verdicts"]["mcc"]["category"],
             "Only qneuro.guard may phrase a quantum advantage claim.")

    log("7/8 checking claims")
    claims = build_claims(ctx)
    note("CLAIMS_CHECKED", [(c["id"], c["status"]) for c in claims], f"{len(claims)} claims", "Each claim gets a status from recomputed values.")

    sp = load_split("withtime")
    perf_e6 = ctx["e6"]["performance_summary"]
    study = _plain(dict(
        meta=dict(
            title="Team Morphos: QSVM survival prediction on heart failure records, re-checked",
            built_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            environment=dict(python=platform.python_version(), numpy=np.__version__, scikit_learn=sklearn.__version__, scipy=scipy.__version__),
            dataset=dict(name="Heart Failure Clinical Records", source="UCI Machine Learning Repository, id 519", patients=299,
                         train=int(len(sp.ytr)), test=int(len(sp.yte)), train_death_rate=round(float(sp.ytr.mean()), 4),
                         test_death_rate=round(float(sp.yte.mean()), 4), test_deaths=int(sp.yte.sum()), split="stratified 80/20, random_state 7, scaler fit on training rows"),
            fast_build=fast,
        ),
        manifest=man,
        e5=ctx["e5"],
        e6=dict(performance_summary=perf_e6, statistical_comparison=ctx["e6"]["statistical_comparison"],
                scale_selection=ctx["e6"]["scale_selection"], compare_with_stored=ctx["e6_compare"]),
        sweeps=dict(e3=ctx.get("e3"), e4=ctx.get("e4")),
        leakage=ctx.get("leakage"),
        gap=ctx.get("gap"),
        sensitivity=ctx["sensitivity"],
        integrity=ctx["integrity"],
        guard=guard,
        claims=claims,
        ledger=_ledger(ctx, claims),
        bloq={tag: json.loads((DATA / "bloq_outputs" / fn).read_text("utf-8")) for tag, fn in
              (("E1", "E1_notime_config.json"), ("E2", "E2_withtime_config.json"))},
        submitted_summary=sub["performance_summary"],
    ))

    log("8/8 signing with ML-DSA-65")
    study_sha = hashlib.sha256(canonical_json(study)).hexdigest()
    note("SIGN_STUDY", dict(study_sha256=study_sha), f"study sha256={study_sha[:16]}", "Seal the study. Any later edit breaks this signature.")
    sig_hex, pk_hex = sign_data(study)
    envelope = dict(
        study=study,
        study_sha256=study_sha,
        signature_hex=sig_hex,
        public_key_hex=pk_hex,
        public_key_fingerprint=get_public_key_fingerprint(pk_hex),
        algorithm="ML-DSA-65",
        audit=dict(entries=chain.length, head_hash=chain.last_hash, file="audit.jsonl"),
    )
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "study.json").write_text(json.dumps(envelope, indent=1), encoding="utf-8")
    shutil.copyfile(chain_path, pub / "audit.jsonl")
    log(f"done: {chain.length} audit entries, fingerprint {envelope['public_key_fingerprint']}")
    return envelope


# ------------------------------------------------------------------ verification
def verify(public_dir: Path | None = None, runs_dir: Path | None = None) -> list[dict[str, Any]]:
    pub = public_dir or PUBLIC
    runs = runs_dir or RUNS
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(dict(name=name, ok=bool(ok), detail=detail))

    env = json.loads((pub / "study.json").read_text("utf-8"))
    study = env["study"]
    add("ML-DSA-65 signature over the study", verify_signature(study, env["signature_hex"], env["public_key_hex"]),
        env["public_key_fingerprint"])
    sha = hashlib.sha256(canonical_json(study)).hexdigest()
    add("Study hash matches the recorded hash", sha == env["study_sha256"], sha[:24] + "...")
    bad = []
    for m in study["manifest"]:
        p = DATA / m["path"]
        if not p.exists() or sha256_file(p) != m["sha256"]:
            bad.append(m["path"])
    add(f"All {len(study['manifest'])} artifacts match the signed manifest", not bad, "mismatch: " + ", ".join(bad) if bad else "every SHA-256 equal")
    ok, idx, reason = verify_chain(runs / "audit.jsonl")
    add("Audit chain (hash links and per-entry signatures)", ok, "intact" if ok else f"fails at entry {idx}: {reason}")
    chain = AuditChain(runs / "audit.jsonl")
    last = chain.entries[-1] if chain.length else None
    add("Last audit entry seals this exact study",
        bool(last and last.action == "SIGN_STUDY" and last.payload_hash == compute_payload_hash(dict(study_sha256=env["study_sha256"]))),
        f"head {chain.last_hash[:24]}...")
    add("Published audit copy equals the local chain", (pub / "audit.jsonl").read_bytes() == (runs / "audit.jsonl").read_bytes(), f"{chain.length} entries")
    return checks


def _flip_digit(data: bytes, anchor: bytes) -> tuple[bytes, int]:
    i = data.find(anchor)
    if i < 0:
        raise ValueError(f"anchor {anchor!r} not found")
    j = data.find(b".", i)
    k = j + 1
    while k < len(data) and chr(data[k]).isdigit():
        k += 1
    pos = k - 1
    b = bytearray(data)
    b[pos] = ord(str((int(chr(b[pos])) + 1) % 10))
    return bytes(b), pos


def tamper_demo(public_dir: Path | None = None) -> list[dict[str, Any]]:
    pub = public_dir or PUBLIC
    env = json.loads((pub / "study.json").read_text("utf-8"))
    out: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        # A: edit one digit of the submitted file's QSVM CV MCC and compare with the signed manifest.
        src = DATA / "submissions" / "SUBMITTED_corrected.json"
        raw = src.read_bytes()
        bad, pos = _flip_digit(raw, b'"mcc_mean"')
        (Path(td) / "SUBMITTED_corrected.json").write_bytes(bad)
        want = next(m["sha256"] for m in env["study"]["manifest"] if m["path"].endswith("SUBMITTED_corrected.json"))
        got = hashlib.sha256(bad).hexdigest()
        before, after = raw[pos - 4:pos + 2].decode(), bad[pos - 4:pos + 2].decode()
        out.append(dict(name="Edit one digit in the submitted file", ok=got == want, expected="FAIL",
                        detail=f"byte {pos}: ...{before} -> ...{after}; signed hash {want[:16]}..., file hash {got[:16]}..."))
        # B: edit one digit inside the signed study.
        raw_env = (pub / "study.json").read_bytes()
        bad_env, pos2 = _flip_digit(raw_env, b'"delta_mcc"')
        tampered = json.loads(bad_env.decode("utf-8"))
        okb = verify_signature(tampered["study"], tampered["signature_hex"], tampered["public_key_hex"])
        out.append(dict(name="Edit one digit inside the signed study", ok=okb, expected="FAIL",
                        detail=f"byte {pos2} changed; ML-DSA-65 signature check returns {okb}"))
        # C: intact copy for contrast.
        out.append(dict(name="Untouched study (control)", ok=verify_signature(env["study"], env["signature_hex"], env["public_key_hex"]), expected="PASS",
                        detail="signature valid"))
    return out
