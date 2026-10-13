"""Pydantic v2 models that define every data boundary in the system."""
from __future__ import annotations

import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    """Whether the dataset was generated or came from a real study."""
    SYNTHETIC = "SYNTHETIC"
    REAL = "REAL"


class ColumnSchema(BaseModel):
    """Schema for a single column — what the agent sees instead of raw data."""
    name: str
    dtype: str
    n_unique: int
    null_count: int


class DatasetSchema(BaseModel):
    """Full dataset schema exposed through the gateway."""
    name: str
    source: DataSource
    n_rows: int
    n_cols: int
    columns: list[ColumnSchema]


class ColumnAggregate(BaseModel):
    """Summary statistics for a numeric column."""
    name: str
    mean: float
    std: float
    min: float
    max: float
    median: float


class DatasetAggregates(BaseModel):
    """Aggregate view of the dataset — the most detail the agent may access."""
    dataset_name: str
    source: DataSource
    numeric_aggregates: list[ColumnAggregate]
    target_distribution: dict[str, int]


class AgentStep(BaseModel):
    """A discrete recorded step taken by the experiment agent."""
    step: int
    action: str
    reason: str


class ModelResult(BaseModel):
    """Metrics from a single model run."""
    model_name: str
    accuracy: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    roc_auc_ci_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    roc_auc_ci_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    roc_auc_folds: list[float] = Field(default_factory=list)
    is_quantum: bool = False
    train_time_seconds: float = Field(ge=0.0)
    params: dict[str, Any] = Field(default_factory=dict)


class GuardVerdict(BaseModel):
    """The guard module's statistical ruling on empirical performance differences."""
    run_id: str
    claim: str
    quantum_better: bool
    metric_used: str = "roc_auc"
    winning_model: str = ""
    best_classical_model: str = ""
    best_quantum_model: str = ""
    classical_best: float
    quantum_best: float
    delta: float
    gap_ci_lower: float | None = None
    gap_ci_upper: float | None = None
    p_value: float | None = None
    p_value_corrected: float | None = None
    confidence_note: str


class BenchmarkRun(BaseModel):
    """Complete record of one benchmark comparison, written to runs/."""
    run_id: str
    timestamp: datetime.datetime
    dataset_name: str
    dataset_source: DataSource
    n_samples: int
    n_features: int
    classical_results: list[ModelResult]
    quantum_results: list[ModelResult]
    agent_steps: list[AgentStep] = Field(default_factory=list)
    verdict: GuardVerdict | None = None
    random_seed: int
    python_version: str
    notes: str = ""


class SignedRun(BaseModel):
    """A benchmark run bundled with its ML-DSA signature for tamper detection."""
    run: BenchmarkRun
    signature_hex: str
    public_key_hex: str
    algorithm: str = "ML-DSA-65"


class AuditEntry(BaseModel):
    """One link in the append-only audit chain."""
    sequence: int
    timestamp: datetime.datetime
    action: str
    run_id: str | None = None
    detail: str = ""
    step_number: int | None = None
    reason: str | None = None
    prev_hash: str
    entry_hash: str
