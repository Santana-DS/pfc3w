"""Audita os artefatos de pré-processamento produzidos pelo experimento R0."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

METADATA_COLUMNS = [
    "instance_id",
    "fault_class",
    "source_type",
    "window_start",
    "window_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Diretório da execução R0 concluída.")
    parser.add_argument(
        "--hash-cleaned",
        action="store_true",
        help="Calcula também o SHA-256 do artefato limpo, que pode ter vários GB.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def audit_features(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    missing_metadata = sorted(set(METADATA_COLUMNS) - set(columns))
    if missing_metadata:
        raise ValueError(f"Colunas de metadados ausentes: {missing_metadata}")

    feature_columns = [column for column in columns if column not in METADATA_COLUMNS]
    null_counts: Counter[str] = Counter()
    nonfinite_counts: Counter[str] = Counter()

    for batch in parquet.iter_batches(columns=feature_columns, batch_size=16_384):
        frame = batch.to_pandas()
        null_counts.update(
            {
                column: int(value)
                for column, value in frame.isna().sum().items()
                if value
            }
        )
        values = frame.to_numpy(dtype=np.float64, copy=False)
        nonfinite = np.logical_not(np.isfinite(values)).sum(axis=0)
        nonfinite_counts.update(
            {
                column: int(value)
                for column, value in zip(feature_columns, nonfinite, strict=True)
                if value
            }
        )

    metadata = pd.read_parquet(path, columns=METADATA_COLUMNS)
    labels_by_instance = metadata.groupby("instance_id", observed=True)["window_label"].nunique()
    sources_by_instance = metadata.groupby("instance_id", observed=True)["source_type"].nunique()

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "n_rows": parquet.metadata.num_rows,
        "n_columns": parquet.metadata.num_columns,
        "n_row_groups": parquet.metadata.num_row_groups,
        "metadata_columns": METADATA_COLUMNS,
        "n_feature_columns": len(feature_columns),
        "feature_columns": feature_columns,
        "n_legacy_instance_ids": int(metadata["instance_id"].nunique()),
        "window_label_counts": sorted_counts(metadata["window_label"]),
        "fault_class_counts": sorted_counts(metadata["fault_class"]),
        "source_counts": sorted_counts(metadata["source_type"]),
        "source_by_window_label": {
            str(source): {
                str(label): int(value)
                for label, value in row.items()
                if value
            }
            for source, row in pd.crosstab(
                metadata["source_type"], metadata["window_label"]
            ).sort_index().iterrows()
        },
        "instances_with_multiple_window_labels": int((labels_by_instance > 1).sum()),
        "instances_with_multiple_sources": int((sources_by_instance > 1).sum()),
        "null_counts": dict(sorted(null_counts.items())),
        "nonfinite_counts": dict(sorted(nonfinite_counts.items())),
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    features_path = run_dir / "features_window_class.parquet"
    cleaned_path = run_dir / "cleaned.parquet"

    with manifest_path.open(encoding="utf-8") as stream:
        run_manifest = json.load(stream)
    if run_manifest.get("status") != "completed":
        raise ValueError("A auditoria exige uma execução concluída.")

    audit = {
        "audited_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_manifest["run_id"],
        "features": audit_features(features_path),
        "cleaned": {
            "path": str(cleaned_path.resolve()),
            "size_bytes": cleaned_path.stat().st_size,
            "sha256": sha256_file(cleaned_path) if args.hash_cleaned else None,
        },
    }

    expected_windows = int(run_manifest["n_windows"])
    actual_windows = int(audit["features"]["n_rows"])
    audit["checks"] = {
        "manifest_window_count_matches": actual_windows == expected_windows,
        "gabriel_documented_window_count_matches": actual_windows == 449_397,
        "gabriel_documented_instance_count_matches": (
            audit["features"]["n_legacy_instance_ids"] == 1_409
        ),
        "all_features_finite": not audit["features"]["nonfinite_counts"],
    }

    output_path = run_dir / "artifact_audit.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
