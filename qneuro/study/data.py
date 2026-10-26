"""Loading the study data bundled in ``data/morphos`` and preparing it from UCI."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "morphos"
VARIANT_FILES = {"notime": "hf_notime", "withtime": "hf_withtime"}
VARIANT_FEATURES = {
    "notime": ["ejection_fraction", "serum_creatinine", "age", "serum_sodium"],
    "withtime": ["ejection_fraction", "serum_creatinine", "age", "time"],
}


@dataclass
class Split:
    variant: str
    columns: list[str]
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte_raw: np.ndarray  # as exported: scaler never saw test rows, so values can leave [0, 1]
    yte: np.ndarray

    @property
    def Xte(self) -> np.ndarray:
        """Test rows clipped to [0, 1], exactly as the study scripts do before encoding."""
        return np.clip(self.Xte_raw, 0, 1)


def load_split(variant: str, data_dir: Path | None = None) -> Split:
    base = (data_dir or DATA) / "original"
    stem = VARIANT_FILES[variant]
    tr = pd.read_csv(base / f"{stem}_train.csv")
    te = pd.read_csv(base / f"{stem}_test.csv")
    cols = list(tr.columns[:-1])
    return Split(
        variant=variant,
        columns=cols,
        Xtr=tr.iloc[:, :-1].values,
        ytr=tr["death_event"].values,
        Xte_raw=te.iloc[:, :-1].values,
        yte=te["death_event"].values,
    )


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_from_uci(out_dir: Path | str, local_csv: Path | str | None = None) -> dict[str, str]:
    """Re-create the four scaled CSVs from the UCI Heart Failure dataset (id 519).

    Same recipe as the hackathon script: one stratified 80/20 patient split with
    random_state=7, MinMaxScaler fit on the training rows only. Returns file -> SHA-256.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if local_csv and Path(local_csv).exists():
        df = pd.read_csv(local_csv)
    else:
        from ucimlrepo import fetch_ucirepo  # imported lazily: only needed for this step

        d = fetch_ucirepo(id=519)
        df = pd.concat([d.data.features, d.data.targets], axis=1)
    df.columns = [c.lower() for c in df.columns]
    train, test = train_test_split(df, test_size=0.2, stratify=df["death_event"], random_state=7)
    hashes: dict[str, str] = {}
    for variant, cols in VARIANT_FEATURES.items():
        sc = MinMaxScaler().fit(train[cols])
        for part, frame in (("train", train), ("test", test)):
            o = pd.DataFrame(sc.transform(frame[cols]), columns=cols)
            o["death_event"] = frame["death_event"].values
            fn = out / f"{VARIANT_FILES[variant]}_{part}.csv"
            o.to_csv(fn, index=False)
            hashes[fn.name] = sha256_file(fn)
    return hashes
