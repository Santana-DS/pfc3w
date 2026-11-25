"""Constrói a tabela real de features retrospectivas compartilhada por E0 e E1."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfc3w.causal import extract_causal_features_from_instance  # noqa: E402
from pfc3w.dataset import parse_instance_identity, sha256_file  # noqa: E402
from pfc3w.features import (  # noqa: E402
    FEATURE_METADATA_COLUMNS,
    attach_feature_identity,
    clean_real_instance,
    expected_feature_columns,
)
from pfc3w.paths import load_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-instances-per-class",
        type=int,
        default=None,
        help=(
            "Número de instâncias válidas por classe no piloto; "
            "omitido processa todas as instâncias reais."
        ),
    )
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument(
        "--variant",
        choices=("retrospective", "causal"),
        default="retrospective",
        help="Transformação temporal usada antes da extração das features.",
    )
    return parser.parse_args()


def git_head(repository: Path) -> str | None:
    process = subprocess.run(
        ["git", f"--git-dir={repository / '.git'}", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def iter_real_files(dataset_root: Path):
    for class_id in range(10):
        files = sorted((dataset_root / str(class_id)).glob("WELL-*.parquet"))
        yield class_id, files


def main() -> None:
    args = parse_args()
    if args.max_instances_per_class is not None and args.max_instances_per_class <= 0:
        raise ValueError("max-instances-per-class precisa ser positivo")
    if args.flush_every <= 0:
        raise ValueError("flush-every precisa ser positivo")

    paths = load_paths(PROJECT_ROOT / "config" / "paths.local.yaml")
    dataset_root = paths["dataset_root"].resolve()
    baseline_repo = paths["baseline_repo"].resolve()
    sys.path.insert(0, str(baseline_repo))
    baseline_config = importlib.import_module("config")
    baseline_features = importlib.import_module("src.feature_engineering")

    sensors = list(baseline_config.KEY_SENSORS)
    suffix = "full" if args.max_instances_per_class is None else "pilot"
    variant_name = "original" if args.variant == "retrospective" else "causal"
    run_id = datetime.now(UTC).strftime(
        f"%Y%m%dT%H%M%SZ_{variant_name}_real_features_{suffix}"
    )
    run_dir = (PROJECT_ROOT / paths["results_root"] / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    features_path = run_dir / f"features_{variant_name}_real.parquet"
    instances_path = run_dir / "instances_processed.csv"
    manifest_path = run_dir / "run_manifest.json"

    with (PROJECT_ROOT / "results" / "inventory" / "data_manifest.json").open(
        encoding="utf-8"
    ) as stream:
        inventory_manifest = json.load(stream)

    manifest: dict[str, object] = {
        "run_id": run_id,
        "status": "running",
        "purpose": (
            "shared_original_real_feature_table_for_E0_E1"
            if args.variant == "retrospective"
            else "shared_causal_real_feature_table_for_E2_E3"
        ),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "project_commit": git_head(PROJECT_ROOT),
        "baseline_commit": git_head(baseline_repo),
        "dataset_version": inventory_manifest["dataset_version"],
        "dataset_fingerprint_sha256": inventory_manifest["dataset_fingerprint_sha256"],
        "parameters": {
            "max_instances_per_class": args.max_instances_per_class,
            "flush_every": args.flush_every,
            "sources": ["REAL"],
            "sensors": sensors,
            "window_size": baseline_config.WINDOW_SIZE,
            "step_size": baseline_config.STEP_SIZE,
            "forward_fill_limit": baseline_config.FFILL_LIMIT,
            "critical_sensor": baseline_config.CRITICAL_SENSOR,
            "max_missing_ratio": baseline_config.MAX_MISSING_RATIO,
            "variant": args.variant,
            "normalization": (
                "full_instance_zscore"
                if args.variant == "retrospective"
                else "expanding_instance_zscore_inclusive"
            ),
            "normalization_ddof": 0,
            "filter": (
                "bidirectional_gaussian"
                if args.variant == "retrospective"
                else "one_sided_gaussian"
            ),
            "gaussian_sigma": baseline_config.GAUSSIAN_SIGMA,
            "gaussian_truncate": 4.0 if args.variant == "causal" else None,
            "decision_time": "window_end",
            "label_strategy": "window_mode",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)

    writer: pq.ParquetWriter | None = None
    buffered_features: list[pd.DataFrame] = []
    records: list[dict[str, object]] = []
    window_labels: Counter[int] = Counter()
    windows_by_class: Counter[int] = Counter()
    wells: set[str] = set()
    n_windows = 0

    def flush() -> None:
        nonlocal writer, n_windows
        if not buffered_features:
            return
        batch = pd.concat(buffered_features, ignore_index=True)
        table = pa.Table.from_pandas(batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(features_path, table.schema, compression="snappy")
        writer.write_table(table)
        n_windows += len(batch)
        buffered_features.clear()

    try:
        for class_id, files in iter_real_files(dataset_root):
            target = args.max_instances_per_class or len(files)
            print(
                f"Classe {class_id}: alvo de {target} instâncias reais válidas "
                f"entre {len(files)} arquivos",
                flush=True,
            )
            processed_in_class = 0
            for position, file_path in enumerate(files, start=1):
                if processed_in_class >= target:
                    break
                identity = parse_instance_identity(file_path, dataset_root)
                relative_path = file_path.relative_to(dataset_root).as_posix()
                frame = pd.read_parquet(file_path)
                n_raw_rows = len(frame)
                cleaned = clean_real_instance(
                    frame,
                    identity,
                    sensors=sensors,
                    ffill_limit=baseline_config.FFILL_LIMIT,
                    critical_sensor=baseline_config.CRITICAL_SENSOR,
                    max_missing_ratio=baseline_config.MAX_MISSING_RATIO,
                )
                del frame

                if cleaned is None:
                    records.append(
                        {
                            "relative_path": relative_path,
                            "instance_id": identity.instance_id,
                            "well_id": identity.well_id,
                            "class_id": identity.class_id,
                            "raw_rows": n_raw_rows,
                            "windows": 0,
                            "status": "discarded_missing_critical_sensor",
                        }
                    )
                    continue

                if args.variant == "retrospective":
                    features = baseline_features.extract_features_from_instance(
                        cleaned,
                        sensors=sensors,
                        window_size=baseline_config.WINDOW_SIZE,
                        step_size=baseline_config.STEP_SIZE,
                        normalize=True,
                        label_strategy="window",
                        smooth_filter="gaussian",
                    )
                else:
                    features = extract_causal_features_from_instance(
                        cleaned,
                        sensors=sensors,
                        window_size=baseline_config.WINDOW_SIZE,
                        step_size=baseline_config.STEP_SIZE,
                        gaussian_sigma=baseline_config.GAUSSIAN_SIGMA,
                    )
                del cleaned
                features = attach_feature_identity(
                    features,
                    identity,
                    relative_path=relative_path,
                    sensors=sensors,
                )
                n_instance_windows = len(features)
                buffered_features.append(features)
                processed_in_class += 1
                wells.add(str(identity.well_id))
                windows_by_class.update({identity.class_id: n_instance_windows})
                window_labels.update(int(value) for value in features["window_label"])
                records.append(
                    {
                        "relative_path": relative_path,
                        "instance_id": identity.instance_id,
                        "well_id": identity.well_id,
                        "class_id": identity.class_id,
                        "raw_rows": n_raw_rows,
                        "windows": n_instance_windows,
                        "status": "processed",
                    }
                )

                if len(buffered_features) >= args.flush_every:
                    flush()
                if position % 50 == 0 or processed_in_class >= target:
                    print(
                        f"  {processed_in_class}/{target} válidas após "
                        f"examinar {position} arquivo(s)",
                        flush=True,
                    )

        flush()
        if writer is None:
            raise RuntimeError("Nenhuma janela foi produzida")
        writer.close()
        writer = None
        pd.DataFrame(records).to_csv(instances_path, index=False)

        processed = [record for record in records if record["status"] == "processed"]
        manifest.update(
            {
                "status": "completed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "outputs": {
                    "features": features_path.name,
                    "features_size_bytes": features_path.stat().st_size,
                    "features_sha256": sha256_file(features_path),
                    "instances": instances_path.name,
                },
                "n_input_instances": len(records),
                "n_processed_instances": len(processed),
                "n_discarded_instances": len(records) - len(processed),
                "n_wells": len(wells),
                "n_windows": n_windows,
                "n_metadata_columns": len(FEATURE_METADATA_COLUMNS),
                "n_feature_columns": len(expected_feature_columns(sensors)),
                "windows_by_fault_class": {
                    str(key): int(value) for key, value in sorted(windows_by_class.items())
                },
                "windows_by_operational_label": {
                    str(key): int(value) for key, value in sorted(window_labels.items())
                },
            }
        )
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    finally:
        if writer is not None:
            writer.close()
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
