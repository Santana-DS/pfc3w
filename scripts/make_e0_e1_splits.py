"""Congela os folds E0 e E1 sobre a tabela real retrospectiva."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.dataset import sha256_file  # noqa: E402
from pfc3w.splits import (  # noqa: E402
    weighted_group_kfold_assignment,
    window_coverage_by_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_run", type=Path)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def git_head(repository: Path) -> str | None:
    process = subprocess.run(
        ["git", f"--git-dir={repository / '.git'}", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def build_split(
    features: pd.DataFrame,
    *,
    group_column: str,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_weights = (
        features.groupby(group_column, as_index=False).size().rename(columns={"size": "n_windows"})
    )
    assignments = weighted_group_kfold_assignment(
        group_weights,
        group_column=group_column,
        weight_column="n_windows",
        n_splits=n_splits,
    )
    coverage = window_coverage_by_fold(
        features,
        assignments,
        group_column=group_column,
        label_column="window_label",
    )
    return assignments, coverage


def split_summary(assignments: pd.DataFrame, coverage: pd.DataFrame) -> dict[str, object]:
    missing_train = coverage.loc[~coverage["train_present"]]
    missing_test = coverage.loc[~coverage["test_present"]]
    return {
        "n_groups": int(len(assignments)),
        "windows_by_fold": {
            str(key): int(value)
            for key, value in assignments.groupby("fold")["n_windows"].sum().items()
        },
        "missing_train_labels": [
            {"fold": int(row.fold), "label": int(row.label)} for row in missing_train.itertuples()
        ],
        "missing_test_labels": [
            {"fold": int(row.fold), "label": int(row.label)} for row in missing_test.itertuples()
        ],
    }


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("n-splits precisa ser pelo menos dois")
    run_dir = args.features_run.resolve()
    features_path = run_dir / "features_original_real.parquet"
    source_manifest_path = run_dir / "run_manifest.json"
    if not features_path.exists():
        raise FileNotFoundError(features_path)

    with source_manifest_path.open(encoding="utf-8") as stream:
        source_manifest = json.load(stream)
    if source_manifest.get("status") != "completed":
        raise ValueError("A extração de features não está concluída")

    output_dir = run_dir / "splits"
    output_dir.mkdir(exist_ok=False)
    features = pd.read_parquet(
        features_path,
        columns=["instance_id", "well_id", "window_label"],
    )

    outputs: dict[str, dict[str, str]] = {}
    summaries: dict[str, object] = {}
    for experiment, group_column in (("E0", "instance_id"), ("E1", "well_id")):
        assignments, coverage = build_split(
            features,
            group_column=group_column,
            n_splits=args.n_splits,
        )
        assignment_path = output_dir / f"{experiment.lower()}_assignments.csv"
        coverage_path = output_dir / f"{experiment.lower()}_coverage.csv"
        assignments.to_csv(assignment_path, index=False)
        coverage.to_csv(coverage_path, index=False)
        outputs[experiment] = {
            "assignments": assignment_path.name,
            "assignments_sha256": sha256_file(assignment_path),
            "coverage": coverage_path.name,
            "coverage_sha256": sha256_file(coverage_path),
        }
        summaries[experiment] = {
            "group_column": group_column,
            **split_summary(assignments, coverage),
        }

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project_commit": git_head(PROJECT_ROOT),
        "source_run_id": source_manifest["run_id"],
        "source_features_sha256": source_manifest["outputs"]["features_sha256"],
        "n_splits": args.n_splits,
        "n_windows": int(len(features)),
        "experiments": summaries,
        "outputs": outputs,
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
