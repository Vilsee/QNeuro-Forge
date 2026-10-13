"""Data access boundary — exposes aggregate metadata without leaking raw instances."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel


class FeatureSummary(BaseModel):
    """Aggregate statistical distribution for a single feature."""
    name: str
    dtype: str
    count: int
    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float


class SchemaInfo(BaseModel):
    """Column definition and null distribution."""
    column_name: str
    dtype: str
    null_count: int
    n_unique: int


class GatewayReport(BaseModel):
    """Permissible exploratory view exposed to the autonomous experiment agent."""
    dataset_source: str
    row_count: int
    feature_count: int
    columns_schema: list[SchemaInfo]
    class_balance: dict[str, int]
    feature_summaries: list[FeatureSummary]


class DatasetGateway:
    """Isolates underlying clinical tabular rows from the agent execution context."""

    def __init__(self, df: pd.DataFrame, source: str, target_col: str = "phenotype_group") -> None:
        # Private reference sequestered inside the instance
        self._df = df.copy()
        self._source = source
        self._target_col = target_col

    def get_row_count(self) -> int:
        """Return total instance count."""
        return len(self._df)

    def get_class_balance(self) -> dict[str, int]:
        """Return target label frequency counts."""
        counts = self._df[self._target_col].value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    def get_schema(self) -> list[SchemaInfo]:
        """Return column names, types, and uniqueness metadata."""
        schema = []
        for col in self._df.columns:
            schema.append(
                SchemaInfo(
                    column_name=col,
                    dtype=str(self._df[col].dtype),
                    null_count=int(self._df[col].isna().sum()),
                    n_unique=int(self._df[col].nunique()),
                )
            )
        return schema

    def get_feature_summaries(self) -> list[FeatureSummary]:
        """Compute aggregate summary statistics per numeric feature."""
        summaries = []
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == self._target_col:
                continue
            s = self._df[col].dropna()
            summaries.append(
                FeatureSummary(
                    name=col,
                    dtype=str(s.dtype),
                    count=int(s.count()),
                    mean=float(s.mean()),
                    std=float(s.std(ddof=1)),
                    min=float(s.min()),
                    p25=float(np.percentile(s, 25)),
                    median=float(np.percentile(s, 50)),
                    p75=float(np.percentile(s, 75)),
                    max=float(s.max()),
                )
            )
        return summaries

    def describe(self) -> GatewayReport:
        """Compile complete permitted metadata report."""
        feature_cols = [c for c in self._df.columns if c != self._target_col]
        return GatewayReport(
            dataset_source=self._source,
            row_count=self.get_row_count(),
            feature_count=len(feature_cols),
            columns_schema=self.get_schema(),
            class_balance=self.get_class_balance(),
            feature_summaries=self.get_feature_summaries(),
        )
