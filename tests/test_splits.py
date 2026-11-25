import pandas as pd

from pfc3w.splits import (
    class_coverage_by_fold,
    estimate_window_count,
    weighted_group_kfold_assignment,
    window_coverage_by_fold,
)


def test_estimate_window_count() -> None:
    assert estimate_window_count(299) == 0
    assert estimate_window_count(300) == 1
    assert estimate_window_count(449) == 1
    assert estimate_window_count(450) == 2


def test_group_assignment_keeps_each_well_in_one_fold() -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["A", "A", "B", "C", "D"],
            "weight": [2, 3, 4, 3, 1],
            "class_id": [0, 1, 0, 1, 0],
        }
    )
    assignments = weighted_group_kfold_assignment(
        frame,
        group_column="well_id",
        weight_column="weight",
        n_splits=2,
    )

    assert assignments["well_id"].is_unique
    assert set(assignments["well_id"]) == {"A", "B", "C", "D"}
    assert assignments.groupby("well_id")["fold"].nunique().max() == 1

    coverage = class_coverage_by_fold(
        frame,
        assignments,
        group_column="well_id",
        class_column="class_id",
        weight_column="weight",
    )
    assert set(coverage.columns) >= {
        "fold",
        "class_id",
        "train_present",
        "test_present",
    }


def test_window_coverage_by_fold_counts_windows_and_groups() -> None:
    frame = pd.DataFrame(
        {
            "well_id": ["A", "A", "B", "B", "C"],
            "window_label": [0, 1, 0, 0, 1],
        }
    )
    assignments = pd.DataFrame({"well_id": ["A", "B", "C"], "fold": [0, 1, 1]})

    coverage = window_coverage_by_fold(
        frame,
        assignments,
        group_column="well_id",
        label_column="window_label",
    )

    row = coverage[(coverage["fold"] == 0) & (coverage["label"] == 0)].iloc[0]
    assert row["test_windows"] == 1
    assert row["train_windows"] == 2
    assert row["test_groups"] == 1
    assert row["train_groups"] == 1
