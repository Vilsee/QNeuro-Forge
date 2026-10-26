"""Tests for the Team Morphos study layer. They re-run real experiments on the bundled data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qneuro.guard import evaluate_advantage
from qneuro.study import experiments as ex
from qneuro.study.build import build, tamper_demo, verify
from qneuro.study.claims import STATUS_LABEL
from qneuro.study.data import DATA, load_split, sha256_file
from qneuro.study.kernels import fidelity_kernel, product_kernel, zz_states
from qneuro.types import FoldScores


def test_split_sizes_and_balance() -> None:
    sp = load_split("withtime")
    assert (len(sp.ytr), len(sp.yte)) == (239, 60)
    assert abs(sp.ytr.mean() - sp.yte.mean()) < 0.02


def test_zz_kernel_is_a_valid_fidelity_kernel() -> None:
    X = load_split("withtime").Xtr[:25]
    S = zz_states(X)
    K = fidelity_kernel(S, S)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert K.min() >= -1e-12 and K.max() <= 1 + 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_product_kernel_matches_pennylane_angle_embedding() -> None:
    qml = pytest.importorskip("pennylane")
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def state(x):
        qml.AngleEmbedding(x, wires=range(4), rotation="X")
        return qml.state()

    X = load_split("withtime").Xtr[:6]
    states = np.array([state(x) for x in X])
    assert np.allclose(fidelity_kernel(states, states), product_kernel(X, X, 1.0), atol=1e-10)


def test_mcc_from_confusion_matches_bloq_output() -> None:
    d = json.loads((DATA / "bloq_outputs" / "E2_withtime_config.json").read_text())
    assert ex.mcc_from_confusion(d["output"]["confusion_matrix"]) == pytest.approx(d["output"]["mcc"], abs=1e-9)
    assert ex.mcc_from_confusion([[33, 0], [15, 0]]) == 0.0


def test_e6_reproduces_the_stored_file_exactly() -> None:
    res = ex.compare_with_stored_e6(ex.benchmark_e6())
    assert res["identical"], res["differences"]


def test_e5_submitted_qsvm_holdout_reproduces() -> None:
    e5 = ex.reproduce_e5()
    for k in ("mcc", "accuracy", "f1", "precision", "recall"):
        assert e5["holdout"][k] == pytest.approx(e5["holdout_submitted"][k], abs=1e-9)
    assert e5["holdout_confusion"] == [[36, 5], [2, 17]]
    assert sum(e5["fold_match"]) >= 4
    assert e5["zz_same_range_holdout_mcc"] != pytest.approx(e5["holdout"]["mcc"], abs=0.01)


def test_logistic_regression_baseline_collapses_when_inputs_are_shrunk() -> None:
    e5 = ex.reproduce_e5()
    assert e5["classical"]["LogisticRegression"]["reproduced"]
    assert e5["logreg_full_range"]["holdout_mcc"] > 0.5


def test_holdout_sensitivity_is_ordered() -> None:
    s = ex.holdout_sensitivity([[36, 5], [2, 17]])
    assert s["two_patients_min"] <= s["one_patient_min"] <= s["baseline"] <= s["one_patient_max"] <= s["two_patients_max"]


def test_guard_sentence_names_the_metric() -> None:
    def fs(name: str, v: list[float]) -> FoldScores:
        a = np.array(v)
        return FoldScores(config_id=name, metric="mcc", scores=v, mean_score=float(a.mean()), std_score=float(a.std(ddof=1)),
                          ci_lower=float(a.mean() - 0.05), ci_upper=float(a.mean() + 0.05))
    base = [0.5, 0.52, 0.48, 0.51, 0.49, 0.5, 0.53]
    v = evaluate_advantage([fs("q", [x + 0.001 for x in base])], [fs("c", base)], metric="mcc")
    assert "MCC" in v.sentence and "AUROC" not in v.sentence


def test_submission_differs_from_original_in_three_text_fields() -> None:
    from qneuro.study.build import _diff

    a = json.loads((DATA / "submissions" / "E5_original_export.json").read_text())
    b = json.loads((DATA / "submissions" / "SUBMITTED_corrected.json").read_text())
    d = _diff(a, b)
    assert len(d) == 3 and all(isinstance(x["before"], str) and isinstance(x["after"], str) for x in d)


def test_build_verify_and_tamper(tmp_path: Path) -> None:
    pub, runs = tmp_path / "pub", tmp_path / "runs"
    env = build(fast=True, public_dir=pub, runs_dir=runs, log=lambda *_: None)
    assert env["algorithm"] == "ML-DSA-65"
    assert all(c["status"] in STATUS_LABEL for c in env["study"]["claims"])
    checks = verify(public_dir=pub, runs_dir=runs)
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]
    for r in tamper_demo(public_dir=pub):
        assert (r["ok"] is False) == (r["expected"] == "FAIL"), r
    # editing the audit chain must break verification at that entry
    lines = (runs / "audit.jsonl").read_text().splitlines()
    rec = json.loads(lines[3]); rec["detail"] = rec["detail"] + "x"; lines[3] = json.dumps(rec)
    (runs / "audit.jsonl").write_text("\n".join(lines) + "\n")
    assert not all(c["ok"] for c in verify(public_dir=pub, runs_dir=runs))


def test_published_study_matches_the_data_files() -> None:
    pub = Path(__file__).resolve().parents[1] / "public" / "data" / "study.json"
    if not pub.exists():
        pytest.skip("run `qneuro study build` first")
    env = json.loads(pub.read_text())
    for m in env["study"]["manifest"]:
        assert sha256_file(DATA / m["path"]) == m["sha256"], m["path"]
