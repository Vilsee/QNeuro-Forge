"""Pydantic v2 domain models shared across all QNeuro Forge modules."""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class VerdictCategory(str, Enum):
    """Categorical classification of statistical arbitration."""
    ADVANTAGE_SUPPORTED = "ADVANTAGE_SUPPORTED"
    NO_DIFFERENCE = "NO_DIFFERENCE"
    CLASSICAL_BETTER = "CLASSICAL_BETTER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ExperimentConfig(BaseModel):
    """Hyperparameter and model configuration for benchmark candidate."""
    model_family: str
    feature_map: str | None = None
    n_qubits: int = 4
    reps: int = 2
    C: float = 1.0
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @property
    def config_id(self) -> str:
        """Deterministic identifier for model configuration."""
        if self.feature_map:
            return f"{self.model_family}_{self.feature_map}_q{self.n_qubits}_r{self.reps}_C{self.C}"
        return f"{self.model_family}_C{self.C}"


class FoldScores(BaseModel):
    """Per-fold evaluation metrics preserving exact fold index correspondence."""
    config_id: str
    metric: str
    scores: list[float]
    mean_score: float
    std_score: float
    ci_lower: float | None = None
    ci_upper: float | None = None


class GuardVerdict(BaseModel):
    """Statistical verdict on model advantage issued exclusively by guard module."""
    category: VerdictCategory
    metric_used: str
    best_quantum_model: str
    best_classical_model: str
    quantum_score: float
    classical_score: float
    delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    configs_tried: int = 1
    sentence: str


class SignedReport(BaseModel):
    """Post-quantum signed research benchmark report."""
    report_id: str
    timestamp: datetime.datetime
    dataset_fingerprint: str
    dataset_source: str
    configs: list[ExperimentConfig]
    fold_scores: list[FoldScores]
    wall_clock_seconds: dict[str, float]
    verdict: GuardVerdict
    signature_hex: str
    public_key_hex: str
    algorithm: str = "ML-DSA-65"
