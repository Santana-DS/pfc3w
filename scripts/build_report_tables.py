"""Gera as tabelas rastreáveis do relatório a partir dos runs congelados."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PRIVATE_REPORT_CONFIG = PROJECT_ROOT / "config" / "reporting" / "e0_e3_tables.yaml"
PUBLIC_REPORT_CONFIG = PROJECT_ROOT / "config" / "reporting" / "public" / "results.yaml"
DEFAULT_REPORT_CONFIG = (
    PRIVATE_REPORT_CONFIG if PRIVATE_REPORT_CONFIG.is_file() else PUBLIC_REPORT_CONFIG
)

from pfc3w.reporting import (  # noqa: E402
    build_e0_e3_tables,
    build_method_and_r0_tables,
    build_support_and_shap_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_REPORT_CONFIG,
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "tables")
    parser.add_argument(
        "--manuscript-tables-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "pfc1" / "latex" / "manuscript" / "tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_e0_e3_tables(
        config_path=args.config.resolve(),
        results_dir=args.results_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        manuscript_tables_dir=args.manuscript_tables_dir.resolve(),
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")
    auxiliary_outputs = build_support_and_shap_tables(
        config_path=args.config.resolve(),
        results_dir=args.results_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        manuscript_tables_dir=args.manuscript_tables_dir.resolve(),
    )
    for kind, path in auxiliary_outputs.items():
        print(f"auxiliary_{kind}: {path}")
    method_outputs = build_method_and_r0_tables(
        config_path=args.config.resolve(),
        results_dir=args.results_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        manuscript_tables_dir=args.manuscript_tables_dir.resolve(),
    )
    for kind, path in method_outputs.items():
        print(f"method_{kind}: {path}")


if __name__ == "__main__":
    main()
