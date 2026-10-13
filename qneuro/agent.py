"""Autonomous experiment agent with multi-provider LLM adapters and deterministic OfflinePlanner."""
from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from qneuro.audit import AuditChain
from qneuro.baselines import _classical_configs, _make_classical_estimator, _ci_95
from qneuro.data import COLUMN_NAMES, TARGET_COLUMN, compute_fingerprint, generate_synthetic
from qneuro.gateway import DatasetGateway, GatewayReport
from qneuro.guard import evaluate_advantage
from qneuro.kernels import compute_cross_kernel, compute_kernel_matrix, fit_fold_scaler
from qneuro.models.schemas import BenchmarkRun, DataSource, ModelResult, AgentStep
from qneuro.types import ExperimentConfig, FoldScores, GuardVerdict, SignedReport
from qneuro.vault import get_or_create_dsa_keys, sign_data, sign_run

MAX_ITERATIONS = 8
MAX_WALL_CLOCK_SECONDS = 180.0
ALLOWED_FEATURE_MAPS = {"angle", "iqp"}
MIN_QUBITS = 4
MAX_QUBITS = 6
MIN_REPS = 1
MAX_REPS = 3
MIN_C = 0.01
MAX_C = 100.0


def validate_config(cfg_dict: dict[str, Any]) -> tuple[bool, str, ExperimentConfig | None]:
    """Validate JSON against ExperimentConfig and strict parameter boundaries."""
    try:
        cfg = ExperimentConfig.model_validate(cfg_dict)
    except Exception as e:
        return False, f"Pydantic validation error: {e}", None

    if cfg.feature_map not in ALLOWED_FEATURE_MAPS:
        return False, f"feature_map must be one of {ALLOWED_FEATURE_MAPS}, got '{cfg.feature_map}'", None

    if not (MIN_QUBITS <= cfg.n_qubits <= MAX_QUBITS):
        return False, f"n_qubits must be between {MIN_QUBITS} and {MAX_QUBITS}, got {cfg.n_qubits}", None

    if not (MIN_REPS <= cfg.reps <= MAX_REPS):
        return False, f"reps must be between {MIN_REPS} and {MAX_REPS}, got {cfg.reps}", None

    if not (MIN_C <= cfg.C <= MAX_C):
        return False, f"C must be between {MIN_C} and {MAX_C}, got {cfg.C}", None

    return True, "", cfg


class LLMPlanner(Protocol):
    """Protocol for experiment proposal generators."""
    def propose_next(
        self,
        report: GatewayReport,
        history: list[tuple[ExperimentConfig, float, tuple[float, float], float]],
        retry_prompt: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        ...


class OfflinePlanner:
    """Deterministic successive-halving planner over quantum configuration space.

    Requires zero network access; guarantees reproducible exploration.
    """

    def __init__(self) -> None:
        # Predefined systematic grid of candidate quantum configs
        self._grid: list[ExperimentConfig] = [
            ExperimentConfig(model_family="quantum_svc", feature_map="angle", n_qubits=4, reps=1, C=1.0),
            ExperimentConfig(model_family="quantum_svc", feature_map="iqp", n_qubits=4, reps=2, C=1.0),
            ExperimentConfig(model_family="quantum_svc", feature_map="angle", n_qubits=6, reps=1, C=1.0),
            ExperimentConfig(model_family="quantum_svc", feature_map="iqp", n_qubits=6, reps=2, C=1.0),
            ExperimentConfig(model_family="quantum_svc", feature_map="iqp", n_qubits=4, reps=3, C=10.0),
            ExperimentConfig(model_family="quantum_svc", feature_map="angle", n_qubits=5, reps=2, C=0.5),
            ExperimentConfig(model_family="quantum_svc", feature_map="iqp", n_qubits=5, reps=1, C=2.0),
            ExperimentConfig(model_family="quantum_svc", feature_map="angle", n_qubits=4, reps=2, C=5.0),
        ]
        self._step = 0

    def propose_next(
        self,
        report: GatewayReport,
        history: list[tuple[ExperimentConfig, float, tuple[float, float], float]],
        retry_prompt: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        idx = self._step % len(self._grid)
        chosen = self._grid[idx]
        self._step += 1
        reason = (
            f"OfflinePlanner systematic exploration step {self._step}: "
            f"evaluating {chosen.feature_map} embedding with {chosen.n_qubits} qubits, "
            f"{chosen.reps} reps, and C={chosen.C}."
        )
        return chosen.model_dump(), reason


class OnlineLLMPlanner:
    """Dispatches reasoning to configured LLM provider via standard HTTP endpoints."""

    def __init__(self, provider: str, api_key: str) -> None:
        self.provider = provider
        self.api_key = api_key

    def _build_prompt(
        self,
        report: GatewayReport,
        history: list[tuple[ExperimentConfig, float, tuple[float, float], float]],
        retry_prompt: str | None = None,
    ) -> str:
        prompt = [
            "You are the autonomous experiment agent for QNeuro Forge.",
            "You must propose the NEXT quantum kernel configuration to test against classical baselines.",
            "",
            "DATASET AGGREGATES (Raw patient rows are strictly sequestered):",
            f"Source: {report.dataset_source}, Samples: {report.row_count}, Features: {report.feature_count}",
            f"Class Balance: {report.class_balance}",
            "",
            "HISTORICAL CONFIGURATIONS TESTED SO FAR:",
        ]
        if not history:
            prompt.append("  (None yet — this is the first iteration)")
        else:
            for cfg, auroc, (ci_l, ci_u), wall in history:
                prompt.append(
                    f"  - {cfg.config_id}: Mean AUROC={auroc:.4f} [95% CI: {ci_l:.4f}, {ci_u:.4f}], Wall-clock={wall:.2f}s"
                )

        prompt.extend([
            "",
            "BOUNDS & REQUIREMENTS:",
            f"- model_family must be 'quantum_svc'",
            f"- feature_map: 'angle' or 'iqp'",
            f"- n_qubits: integer between {MIN_QUBITS} and {MAX_QUBITS}",
            f"- reps: integer between {MIN_REPS} and {MAX_REPS}",
            f"- C: float between {MIN_C} and {MAX_C}",
            "",
            "Output MUST be valid JSON with two keys:",
            "1. 'config': object matching the required fields",
            "2. 'reason': a plain-English explanation of your scientific hypothesis for this step",
        ])

        if retry_prompt:
            prompt.append(f"\nPREVIOUS ERROR TO CORRECT: {retry_prompt}")

        return "\n".join(prompt)

    def propose_next(
        self,
        report: GatewayReport,
        history: list[tuple[ExperimentConfig, float, tuple[float, float], float]],
        retry_prompt: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        prompt_str = self._build_prompt(report, history, retry_prompt)

        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_str}]}],
                "generationConfig": {"response_mime_type": "application/json"},
            }
            res = httpx.post(url, json=payload, timeout=30.0)
            res.raise_for_status()
            text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif self.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt_str}],
            }
            res = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            res.raise_for_status()
            text = res.json()["content"][0]["text"]
        else:  # openai
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {
                "model": "gpt-4o-mini",
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt_str}],
            }
            res = httpx.post(url, headers=headers, json=payload, timeout=30.0)
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"]

        parsed = json.loads(text)
        return parsed.get("config", parsed), parsed.get("reason", "LLM proposed step")


def select_planner() -> LLMPlanner:
    """Select appropriate planner based on environment variables."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if gemini_key:
        return OnlineLLMPlanner("gemini", gemini_key)
    if anthropic_key:
        return OnlineLLMPlanner("anthropic", anthropic_key)
    if openai_key:
        return OnlineLLMPlanner("openai", openai_key)

    return OfflinePlanner()


def run_agent_pipeline(
    n_samples: int = 120,
    random_seed: int = 42,
    budget: int = 5,
    runs_dir: Path | str | None = None,
    audit_path: Path | str | None = None,
) -> tuple[SignedReport, Path]:
    """Execute end-to-end autonomous research benchmark with post-quantum signed report."""
    start_time = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]

    if runs_dir is None:
        runs_dir = Path("runs")
    else:
        runs_dir = Path(runs_dir)

    target_run_dir = runs_dir / run_id
    target_run_dir.mkdir(parents=True, exist_ok=True)

    if audit_path is None:
        audit_path = target_run_dir / "audit.jsonl"
    else:
        audit_path = Path(audit_path)

    audit = AuditChain(audit_path)

    # 1. Synthesize cohort & initialize gateway
    df, source_label = generate_synthetic(n_samples=n_samples, random_seed=random_seed)
    fingerprint = compute_fingerprint(df)
    gateway = DatasetGateway(df, source=source_label, target_col=TARGET_COLUMN)
    report_metadata = gateway.describe()

    audit.append(
        action="INITIALIZE_GATEWAY",
        run_id=run_id,
        detail=f"Sequestered {report_metadata.row_count} rows. Features: {report_metadata.feature_count}",
        reason="Dataset privacy boundary strictly isolates raw clinical records.",
    )

    planner = select_planner()
    offline_fallback = OfflinePlanner()

    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X = df[COLUMN_NAMES].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_seed)
    folds = list(rskf.split(X, y))

    quantum_fold_scores: list[FoldScores] = []
    quantum_configs_tried: list[ExperimentConfig] = []
    wall_clocks: dict[str, float] = {}
    history: list[tuple[ExperimentConfig, float, tuple[float, float], float]] = []

    iterations_to_run = min(budget, MAX_ITERATIONS)

    # 2. Iterative optimization loop
    for step in range(1, iterations_to_run + 1):
        if (time.perf_counter() - start_time) > MAX_WALL_CLOCK_SECONDS:
            audit.append(action="BUDGET_EXCEEDED", run_id=run_id, detail="Wall-clock limit reached.")
            break

        # Propose next config
        retry_msg: str | None = None
        valid_cfg: ExperimentConfig | None = None
        stated_reason: str = ""

        for attempt in range(2):
            try:
                raw_cfg, stated_reason = planner.propose_next(report_metadata, history, retry_prompt=retry_msg)
                is_valid, err_msg, parsed_cfg = validate_config(raw_cfg)
                if is_valid and parsed_cfg is not None:
                    valid_cfg = parsed_cfg
                    break
                else:
                    retry_msg = err_msg
            except Exception as e:
                retry_msg = f"Proposal failure: {e}"

        if valid_cfg is None:
            # Fall back to offline planner
            raw_cfg, stated_reason = offline_fallback.propose_next(report_metadata, history)
            _, _, valid_cfg = validate_config(raw_cfg)
            stated_reason += " (Fallback from failed online proposal)"

        assert valid_cfg is not None
        quantum_configs_tried.append(valid_cfg)

        # Audit proposal
        audit.append(
            action="PROPOSE_QUANTUM_CONFIG",
            run_id=run_id,
            step_number=step,
            detail=valid_cfg.config_id,
            reason=stated_reason,
        )

        # Evaluate proposed config on folds
        t0 = time.perf_counter()
        fold_aurocs: list[float] = []
        fold_f1s: list[float] = []
        fold_accs: list[float] = []

        for train_idx, test_idx in folds:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = fit_fold_scaler(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            K_train = compute_kernel_matrix(X_tr_s, valid_cfg, data_fingerprint=fingerprint, use_cache=True)
            K_test = compute_cross_kernel(X_te_s, X_tr_s, valid_cfg, data_fingerprint=fingerprint, use_cache=True)

            clf = SVC(kernel="precomputed", C=valid_cfg.C, probability=True, random_state=random_seed)
            clf.fit(K_train, y_tr)

            y_score = clf.predict_proba(K_test)[:, 1]
            y_pred = clf.predict(K_test)

            fold_aurocs.append(float(roc_auc_score(y_te, y_score)))
            fold_f1s.append(float(f1_score(y_te, y_pred, zero_division=0.0)))
            fold_accs.append(float(accuracy_score(y_te, y_pred)))

        elapsed = time.perf_counter() - t0
        cid = valid_cfg.config_id
        wall_clocks[cid] = elapsed

        arr_auc = np.array(fold_aurocs)
        ci_l, ci_u = _ci_95(arr_auc)
        history.append((valid_cfg, float(arr_auc.mean()), (ci_l, ci_u), elapsed))

        # Record FoldScores
        fs_auroc = FoldScores(
            config_id=cid,
            metric="auroc",
            scores=fold_aurocs,
            mean_score=float(arr_auc.mean()),
            std_score=float(arr_auc.std(ddof=1)),
            ci_lower=ci_l,
            ci_upper=ci_u,
        )
        quantum_fold_scores.append(fs_auroc)

        audit.append(
            action="EVALUATE_CONFIG",
            run_id=run_id,
            step_number=step,
            detail=f"{cid}: mean_auroc={arr_auc.mean():.4f} time={elapsed:.2f}s",
            reason="RepeatedStratifiedKFold evaluation complete.",
        )

    # 3. Evaluate classical baselines
    classical_fold_scores: list[FoldScores] = []
    for ccfg, pgrid in _classical_configs():
        t0 = time.perf_counter()
        c_aurocs = []
        for train_idx, test_idx in folds:
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            est = _make_classical_estimator(ccfg)
            est.fit(X_tr_s, y_tr)
            y_score = est.predict_proba(X_te_s)[:, 1]
            c_aurocs.append(float(roc_auc_score(y_te, y_score)))

        elapsed = time.perf_counter() - t0
        cid = ccfg.config_id
        wall_clocks[cid] = elapsed
        arr = np.array(c_aurocs)
        ci_l, ci_u = _ci_95(arr)
        classical_fold_scores.append(
            FoldScores(
                config_id=cid,
                metric="auroc",
                scores=c_aurocs,
                mean_score=float(arr.mean()),
                std_score=float(arr.std(ddof=1)),
                ci_lower=ci_l,
                ci_upper=ci_u,
            )
        )

    # 4. Guard arbitration
    verdict = evaluate_advantage(quantum_fold_scores, classical_fold_scores, metric="auroc")
    audit.append(
        action="GUARD_ARBITRATION",
        run_id=run_id,
        detail=verdict.sentence,
        reason="Nadeau-Bengio corrected resampled t-test with Holm family-wise correction.",
    )

    # 5. Compile and sign report
    all_configs = quantum_configs_tried + [cfg for cfg, _ in _classical_configs()]
    all_scores = quantum_fold_scores + classical_fold_scores

    report_data = {
        "report_id": run_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset_fingerprint": fingerprint,
        "dataset_source": source_label,
        "configs": [c.model_dump() for c in all_configs],
        "fold_scores": [s.model_dump() for s in all_scores],
        "wall_clock_seconds": wall_clocks,
        "verdict": verdict.model_dump(),
    }
    sk, pk = get_or_create_dsa_keys()
    sig_hex, pk_hex = sign_data(report_data, sk=sk, pk=pk)

    signed_report = SignedReport(
        report_id=run_id,
        timestamp=datetime.datetime.fromisoformat(report_data["timestamp"]),
        dataset_fingerprint=fingerprint,
        dataset_source=source_label,
        configs=all_configs,
        fold_scores=all_scores,
        wall_clock_seconds=wall_clocks,
        verdict=verdict,
        signature_hex=sig_hex,
        public_key_hex=pk_hex,
        algorithm="ML-DSA-65",
    )

    report_file = target_run_dir / "report.json"
    report_file.write_text(signed_report.model_dump_json(indent=2), encoding="utf-8")

    # Also save a top-level run file for backward compatibility with runs/<id>.json
    top_run_file = runs_dir / f"{run_id}.json"
    top_run_file.write_text(signed_report.model_dump_json(indent=2), encoding="utf-8")

    audit.append(
        action="NOTARISE_REPORT",
        run_id=run_id,
        detail=f"Signed report notarized at {report_file} with ML-DSA-65.",
        reason="Post-quantum cryptographic provenance sealed.",
    )

    return signed_report, target_run_dir


# Legacy function bridge
def run_experiment(
    random_seed: int = 42,
    n_samples: int = 80,
    runs_dir: Path | None = None,
    audit_path: Path | None = None,
) -> tuple[Any, Any]:
    """Compatibility runner for legacy callers."""
    from qneuro.bench import run_benchmark
    from qneuro.synth import generate
    from qneuro.guard import evaluate

    if runs_dir is None:
        runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    if audit_path is None:
        audit_path = runs_dir / "audit.jsonl"

    audit = AuditChain(audit_path)
    df, source = generate(n_samples=n_samples, random_seed=random_seed)
    X = df.drop(columns=["diagnosis"]).to_numpy()
    y = df["diagnosis"].to_numpy()

    bench_run = run_benchmark(
        X=X,
        y=y,
        dataset_name="synth_neurodev",
        dataset_source=source,
        random_seed=random_seed,
        n_folds=5,
        runs_dir=runs_dir,
    )
    verdict = evaluate(bench_run, metric="roc_auc")
    bench_run.verdict = verdict
    signed = sign_run(bench_run)
    out_file = runs_dir / f"{bench_run.run_id}.json"
    out_file.write_text(signed.model_dump_json(indent=2), encoding="utf-8")
    return signed, verdict


def main() -> None:
    print("=" * 80)
    print("QNEURO FORGE — AUTONOMOUS EXPERIMENT PIPELINE")
    print("=" * 80)
    report, run_dir = run_agent_pipeline(budget=4)
    print(f"\nRun ID: {report.report_id}")
    print(f"Dataset: {report.dataset_source} (Fingerprint: {report.dataset_fingerprint[:16]}...)")
    print(f"Artifacts saved to: {run_dir}")
    print("\n=== Classical Baselines ===")
    for fs in report.fold_scores:
        if "quantum" not in fs.config_id and fs.metric == "auroc":
            print(f"  {fs.config_id:<35} AUROC={fs.mean_score:.4f} [{fs.ci_lower:.4f}, {fs.ci_upper:.4f}]")

    print("\n=== Quantum Configurations Explored ===")
    for fs in report.fold_scores:
        if "quantum" in fs.config_id and fs.metric == "auroc":
            print(f"  {fs.config_id:<35} AUROC={fs.mean_score:.4f} [{fs.ci_lower:.4f}, {fs.ci_upper:.4f}]")

    print("\n=== Guard Verdict ===")
    print(f"Category: {report.verdict.category.value}")
    print(f"Delta: {report.verdict.delta:+.4f} (95% CI: [{report.verdict.ci_lower:+.4f}, {report.verdict.ci_upper:+.4f}])")
    print(f"Nadeau-Bengio Corrected p-value: {report.verdict.p_value:.4f} (across {report.verdict.configs_tried} configs)")
    print(f"Ruling: {report.verdict.sentence}")
    print(f"\nSignature: {report.signature_hex[:32]}... ({report.algorithm})")

    # Verify audit chain
    audit_file = run_dir / "audit.jsonl"
    ac = AuditChain(audit_file)
    valid, fail_idx, reason = ac.verify_chain()
    print(f"Audit Chain: {ac.length} entries. Integrity Valid: {valid}")


if __name__ == "__main__":
    main()
