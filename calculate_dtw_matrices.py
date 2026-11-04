#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import dtw as DTW
from scipy.spatial.distance import euclidean


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "time_series" / "alibaba" 
OUTPUT_DIR = BASE_DIR / "other-results" / "dtw"

MICROSERVICES = range(1, 11)

METRICS = {
    "cpu": "cpu.csv",
    "memory": "memory.csv",
    "response_time": "response_time.csv",
    "traffic": "traffic.csv",
}


# ---------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------

def load_series(csv_path: Path) -> np.ndarray:
    """Load one univariate time series from a CSV file."""

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)

    if "value" in frame.columns:
        values = pd.to_numeric(frame["value"], errors="coerce")
    else:
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            raise ValueError(
                f"No 'value' column or numeric column found in {csv_path}"
            )
        values = pd.to_numeric(frame[numeric_columns[0]], errors="coerce")

    if values.isna().any():
        bad_rows = values[values.isna()].index.tolist()[:10]
        raise ValueError(
            f"Missing/non-numeric observations in {csv_path}; "
            f"first affected row indices: {bad_rows}"
        )

    series = values.to_numpy(dtype=float)

    if series.ndim != 1 or len(series) == 0:
        raise ValueError(f"Expected a non-empty univariate series: {csv_path}")

    return series


# ---------------------------------------------------------------------
# DTW: follows the recovered implementation
# ---------------------------------------------------------------------

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)

    result = DTW.dtw(a, b, dist=euclidean)
    return float(result[0])


# ---------------------------------------------------------------------
# Pairwise matrix
# ---------------------------------------------------------------------

def calculate_metric_matrix(metric_name: str, filename: str) -> pd.DataFrame:
    """Calculate the 10 x 10 DTW matrix for one metric."""

    series_by_microservice = {}

    for microservice_id in MICROSERVICES:
        csv_path = (
            DATA_DIR
            / f"microservice {microservice_id}"
            / filename
        )
        print(f"  Loading: {csv_path}", flush=True)
        series_by_microservice[microservice_id] = load_series(csv_path)

    ids = list(MICROSERVICES)
    labels = [f"microservice {i}" for i in ids]
    matrix = np.zeros((len(ids), len(ids)), dtype=float)

    # Compute each unordered pair once; mirror the result because DTW
    # distance is symmetric for this use.
    total_pairs = len(ids) * (len(ids) - 1) // 2
    pair_number = 0

    for i, id_a in enumerate(ids):
        for j in range(i + 1, len(ids)):
            id_b = ids[j]
            pair_number += 1
            print(
                f"  [{pair_number}/{total_pairs}] "
                f"DTW(microservice {id_a}, microservice {id_b})",
                flush=True,
            )

            distance = dtw_distance(
                series_by_microservice[id_a],
                series_by_microservice[id_b],
            )

            matrix[i, j] = distance
            matrix[j, i] = distance

    return pd.DataFrame(matrix, index=labels, columns=labels)


def main() -> int:
    print(f"Data directory: {DATA_DIR}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}", flush=True)

    if not DATA_DIR.is_dir():
        print(
            f"ERROR: expected data directory does not exist: {DATA_DIR}",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for metric_name, filename in METRICS.items():
        print(f"\nCalculating DTW matrix for {metric_name}...", flush=True)

        matrix = calculate_metric_matrix(metric_name, filename)

        output_path = OUTPUT_DIR / f"{metric_name}.csv"
        matrix.to_csv(output_path, float_format="%.12g")

        print(f"Saved: {output_path}", flush=True)
        print(matrix.round(6).to_string(), flush=True)

    print("\nAll four DTW matrices were generated.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
