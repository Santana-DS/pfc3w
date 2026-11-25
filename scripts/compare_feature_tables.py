"""Compara estrutural e numericamente duas tabelas Parquet de features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.dataset import sha256_file  # noqa: E402
from pfc3w.features import FEATURE_METADATA_COLUMNS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def compare_tables(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    """Compara tabelas por row group para limitar o uso de memória."""
    reference = pq.ParquetFile(reference_path)
    candidate = pq.ParquetFile(candidate_path)
    reference_columns = reference.schema_arrow.names
    candidate_columns = candidate.schema_arrow.names
    feature_columns = [
        column for column in reference_columns if column not in FEATURE_METADATA_COLUMNS
    ]

    schema_equal = reference.schema_arrow.equals(candidate.schema_arrow)
    row_groups_compatible = (
        reference.num_row_groups == candidate.num_row_groups
        and all(
            reference.metadata.row_group(index).num_rows
            == candidate.metadata.row_group(index).num_rows
            for index in range(min(reference.num_row_groups, candidate.num_row_groups))
        )
    )
    metadata_equal = schema_equal and row_groups_compatible
    missing_pattern_equal = schema_equal and row_groups_compatible
    finite_values_equal = schema_equal and row_groups_compatible
    changed_comparable_finite_cells = 0
    different_cells_total = 0
    comparable_finite_cells = 0
    absolute_difference_sum = 0.0
    maximum_absolute_difference = 0.0
    reference_missing_candidate_finite = np.zeros(len(feature_columns), dtype=np.int64)
    reference_finite_candidate_missing = np.zeros(len(feature_columns), dtype=np.int64)

    if schema_equal and row_groups_compatible:
        for row_group in range(reference.num_row_groups):
            reference_metadata = reference.read_row_group(
                row_group, columns=list(FEATURE_METADATA_COLUMNS)
            ).to_pandas()
            candidate_metadata = candidate.read_row_group(
                row_group, columns=list(FEATURE_METADATA_COLUMNS)
            ).to_pandas()
            metadata_equal &= reference_metadata.equals(candidate_metadata)

            reference_features = reference.read_row_group(
                row_group, columns=feature_columns
            ).to_pandas().to_numpy(dtype=float)
            candidate_features = candidate.read_row_group(
                row_group, columns=feature_columns
            ).to_pandas().to_numpy(dtype=float)
            reference_missing = np.isnan(reference_features)
            candidate_missing = np.isnan(candidate_features)
            missing_pattern_equal &= np.array_equal(reference_missing, candidate_missing)
            reference_missing_candidate_finite += (
                reference_missing & ~candidate_missing
            ).sum(axis=0)
            reference_finite_candidate_missing += (
                ~reference_missing & candidate_missing
            ).sum(axis=0)

            comparable = ~reference_missing & ~candidate_missing
            differences = np.abs(
                reference_features[comparable] - candidate_features[comparable]
            )
            comparable_finite_cells += int(comparable.sum())
            different_cells_total += int(
                (~np.isclose(reference_features, candidate_features, equal_nan=True)).sum()
            )
            changed_comparable_finite_cells += int(
                (~np.isclose(reference_features[comparable], candidate_features[comparable])).sum()
            )
            if differences.size:
                absolute_difference_sum += float(differences.sum())
                maximum_absolute_difference = max(
                    maximum_absolute_difference,
                    float(differences.max()),
                )
                finite_values_equal &= bool(np.all(differences == 0))

    return {
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": sha256_file(reference_path),
            "rows": reference.metadata.num_rows,
            "columns": len(reference_columns),
            "row_groups": reference.num_row_groups,
        },
        "candidate": {
            "path": str(candidate_path.resolve()),
            "sha256": sha256_file(candidate_path),
            "rows": candidate.metadata.num_rows,
            "columns": len(candidate_columns),
            "row_groups": candidate.num_row_groups,
        },
        "schema_equal": schema_equal,
        "row_groups_compatible": row_groups_compatible,
        "metadata_equal": metadata_equal,
        "missing_pattern_equal": missing_pattern_equal,
        "finite_values_equal": finite_values_equal,
        "comparable_finite_cells": comparable_finite_cells,
        "different_cells_total": different_cells_total,
        "changed_comparable_finite_cells": changed_comparable_finite_cells,
        "reference_missing_candidate_finite": int(reference_missing_candidate_finite.sum()),
        "reference_finite_candidate_missing": int(reference_finite_candidate_missing.sum()),
        "missing_mismatches_by_feature": {
            feature: {
                "reference_missing_candidate_finite": int(
                    reference_missing_candidate_finite[index]
                ),
                "reference_finite_candidate_missing": int(
                    reference_finite_candidate_missing[index]
                ),
            }
            for index, feature in enumerate(feature_columns)
            if reference_missing_candidate_finite[index]
            or reference_finite_candidate_missing[index]
        },
        "mean_absolute_difference": (
            absolute_difference_sum / comparable_finite_cells
            if comparable_finite_cells
            else None
        ),
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def main() -> None:
    args = parse_args()
    payload = compare_tables(args.reference.resolve(), args.candidate.resolve())
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
