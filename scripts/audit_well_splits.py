"""Audita folds fisicamente independentes antes do treinamento."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.splits import (  # noqa: E402
    class_coverage_by_fold,
    estimate_window_count,
    weighted_group_kfold_assignment,
)
from pfc3w.validation import assert_disjoint_groups  # noqa: E402


def main() -> None:
    inventory_path = PROJECT_ROOT / "results" / "inventory" / "instances.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventário ausente: {inventory_path}. Execute scripts/build_inventory.py primeiro."
        )

    inventory = pd.read_csv(inventory_path)
    real = inventory.loc[inventory["source_type"] == "REAL"].copy()
    real["estimated_windows"] = real["num_rows"].map(estimate_window_count)

    assignments = weighted_group_kfold_assignment(
        real,
        group_column="well_id",
        weight_column="estimated_windows",
        n_splits=5,
    )
    coverage = class_coverage_by_fold(
        real,
        assignments,
        group_column="well_id",
        class_column="class_id",
        weight_column="estimated_windows",
    )

    summaries = []
    all_classes = sorted(int(value) for value in real["class_id"].unique())
    all_wells = set(real["well_id"])
    for fold in sorted(assignments["fold"].unique()):
        test_wells = set(assignments.loc[assignments["fold"] == fold, "well_id"])
        train_wells = all_wells - test_wells
        assert_disjoint_groups(train_wells, test_wells)

        fold_coverage = coverage.loc[coverage["fold"] == fold]
        test_classes = sorted(
            int(value)
            for value in fold_coverage.loc[fold_coverage["test_present"], "class_id"]
        )
        train_missing = sorted(
            int(value)
            for value in fold_coverage.loc[~fold_coverage["train_present"], "class_id"]
        )
        summaries.append(
            {
                "fold": int(fold),
                "test_wells": sorted(test_wells),
                "n_test_wells": len(test_wells),
                "estimated_train_windows": int(
                    assignments.loc[assignments["fold"] != fold, "estimated_windows"].sum()
                ),
                "estimated_test_windows": int(
                    assignments.loc[assignments["fold"] == fold, "estimated_windows"].sum()
                ),
                "test_classes": test_classes,
                "test_missing_classes": sorted(set(all_classes) - set(test_classes)),
                "train_missing_classes": train_missing,
            }
        )

    output_dir = PROJECT_ROOT / "results" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "well_groupkfold5_assignments.csv", index=False)
    coverage.to_csv(output_dir / "well_groupkfold5_class_coverage.csv", index=False)

    logo = assignments[["well_id", "estimated_windows"]].sort_values("well_id").copy()
    logo["fold"] = range(len(logo))
    logo.to_csv(output_dir / "leave_one_well_out_assignments.csv", index=False)

    report = {
        "scope": "real_instances_only",
        "window_count_is_structural_estimate": True,
        "window_size": 300,
        "step_size": 150,
        "n_wells": int(real["well_id"].nunique()),
        "n_instances": int(len(real)),
        "estimated_windows": int(real["estimated_windows"].sum()),
        "groupkfold_n_splits": 5,
        "leave_one_well_out_n_splits": int(real["well_id"].nunique()),
        "folds": summaries,
    }
    with (output_dir / "split_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
