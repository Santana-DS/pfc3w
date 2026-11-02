"""Gera manifesto e tabelas de cobertura do 3W Dataset."""

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

from pfc3w.dataset import (  # noqa: E402
    dataset_fingerprint,
    inspect_instance,
    iter_parquet_files,
    read_dataset_version,
)
from pfc3w.paths import load_paths  # noqa: E402


def git_head(repository: Path) -> str | None:
    """Obtém o commit sem depender de alterar o diretório corrente."""
    git_dir = repository / ".git"
    if not git_dir.exists():
        return None
    process = subprocess.run(
        ["git", f"--git-dir={git_dir}", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths",
        type=Path,
        default=PROJECT_ROOT / "config" / "paths.local.yaml",
        help="Arquivo local de caminhos.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Calcula SHA-256 de todos os Parquet (mais lento, mas reprodutível).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths)
    dataset_root = paths["dataset_root"].resolve()
    output_dir = (PROJECT_ROOT / paths["results_root"] / "inventory").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_parquet_files(dataset_root))
    records = []
    for index, file_path in enumerate(files, start=1):
        records.append(inspect_instance(file_path, dataset_root, include_hash=args.hash))
        if index % 250 == 0 or index == len(files):
            print(f"Inventariados {index}/{len(files)} arquivos", flush=True)

    frame = pd.DataFrame(record.to_dict() for record in records)
    frame.to_csv(output_dir / "instances.csv", index=False)

    class_source = (
        frame.groupby(["class_id", "source_type"], dropna=False)
        .agg(instances=("instance_id", "nunique"), rows=("num_rows", "sum"))
        .reset_index()
    )
    class_source.to_csv(output_dir / "class_by_source.csv", index=False)

    real = frame.loc[frame["source_type"] == "REAL"]
    class_well = pd.crosstab(real["well_id"], real["class_id"])
    class_well.to_csv(output_dir / "class_by_well.csv")

    duplicate_legacy_ids = int(frame["legacy_instance_id"].duplicated(keep=False).sum())
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": read_dataset_version(dataset_root),
        "dataset_repository_commit": git_head(paths["dataset_repo"].resolve()),
        "baseline_repository_commit": git_head(paths["baseline_repo"].resolve()),
        "hashes_included": args.hash,
        "dataset_fingerprint_sha256": dataset_fingerprint(records) if args.hash else None,
        "n_files": int(len(frame)),
        "size_bytes": int(frame["size_bytes"].sum()),
        "n_rows": int(frame["num_rows"].sum()),
        "n_real_wells": int(real["well_id"].nunique()),
        "n_instances_by_source": {
            str(key): int(value) for key, value in frame["source_type"].value_counts().items()
        },
        "n_instances_by_class": {
            str(key): int(value)
            for key, value in frame["class_id"].value_counts().sort_index().items()
        },
        "n_duplicate_rows_by_legacy_instance_id": duplicate_legacy_ids,
        "outputs": {
            "instances": "instances.csv",
            "class_by_source": "class_by_source.csv",
            "class_by_well": "class_by_well.csv",
        },
    }
    with (output_dir / "data_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
