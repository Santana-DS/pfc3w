"""Geração rastreável de tabelas acadêmicas a partir de runs imutáveis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

METRIC_COLUMNS = (
    "f1_macro_all_labels",
    "f1_macro_physically_identifiable",
    "f1_weighted",
    "balanced_accuracy",
    "accuracy",
)


def _source_run_id(specification: dict[str, Any]) -> str:
    """Retorna o nome local do diretório legado que contém os artefatos."""
    return str(specification.get("source_run_id", specification["run_id"]))


def _published_run_id(specification: dict[str, Any]) -> str:
    """Retorna a identidade estável exposta em tabelas e proveniência."""
    return str(specification.get("scientific_run_id", _source_run_id(specification)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not config.get("runs") or not config.get("contrasts"):
        raise ValueError("Configuração deve conter runs e contrasts")
    return config


def collect_metric_rows(results_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    dataset_signature: tuple[int, tuple[int, ...]] | None = None

    for specification in config["runs"]:
        source_run_id = _source_run_id(specification)
        published_run_id = _published_run_id(specification)
        run_dir = results_dir / source_run_id
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "metrics.json"
        if not manifest_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Artefatos ausentes no run {source_run_id}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        experiment_id = str(specification["experiment_id"])
        model = str(specification["model"])
        pair = (model, experiment_id)
        if pair in seen_pairs:
            raise ValueError(f"Par modelo-experimento duplicado: {pair}")
        seen_pairs.add(pair)

        if manifest.get("status") != "completed":
            raise ValueError(f"Run não concluído: {source_run_id}")
        if manifest.get("experiment_id") != experiment_id or manifest.get("model") != model:
            raise ValueError(f"Identidade divergente no manifesto de {source_run_id}")

        dataset = manifest.get("dataset", {})
        signature = (int(dataset["n_windows"]), tuple(dataset["all_labels"]))
        if dataset_signature is None:
            dataset_signature = signature
        elif signature != dataset_signature:
            raise ValueError(f"População incompatível no run {source_run_id}")

        metrics = payload.get("metrics", {})
        if any(column not in metrics for column in METRIC_COLUMNS):
            raise ValueError(f"Métrica obrigatória ausente no run {source_run_id}")
        for column in METRIC_COLUMNS:
            if float(metrics[column]) != float(manifest["metrics"][column]):
                raise ValueError(
                    f"Métrica {column} diverge do manifesto de {source_run_id}"
                )

        rows.append(
            {
                "model": model,
                "display_model": str(specification["display_model"]),
                "experiment_id": experiment_id,
                **{column: float(metrics[column]) for column in METRIC_COLUMNS},
                "run_id": published_run_id,
                "_source_run_id": source_run_id,
                "project_commit": manifest.get("project_commit"),
                "metrics_sha256": sha256_file(metrics_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    return rows


def calculate_contrasts(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    by_pair = {(row["model"], row["experiment_id"]): row for row in rows}
    model_order = list(dict.fromkeys(row["model"] for row in rows))
    output: list[dict[str, Any]] = []
    for specification in config["contrasts"]:
        values: dict[str, float] = {}
        run_ids: dict[str, str] = {}
        for model in model_order:
            minuend = by_pair[(model, str(specification["minuend"]))]
            subtrahend = by_pair[(model, str(specification["subtrahend"]))]
            values[model] = round(
                minuend["f1_macro_all_labels"] - subtrahend["f1_macro_all_labels"], 4
            )
            run_ids[model] = f"{minuend['run_id']};{subtrahend['run_id']}"
        output.append(
            {
                "contrast_id": str(specification["contrast_id"]),
                "factor_latex": str(specification["factor_latex"]),
                "values": values,
                "run_ids": run_ids,
            }
        )
    return output


def collect_per_well_summaries(
    results_dir: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resume métricas dando o mesmo peso a cada poço físico."""
    summaries: list[dict[str, Any]] = []
    population_by_well: dict[str, tuple[int, int]] | None = None
    required_columns = {
        "well_id",
        "n_windows",
        "n_labels",
        "f1_macro_present_labels",
        "accuracy",
    }

    for row in rows:
        run_id = row["run_id"]
        source_run_id = row.get("_source_run_id", run_id)
        run_dir = results_dir / source_run_id
        per_well_path = run_dir / "per_well_metrics.csv"
        manifest_path = run_dir / "run_manifest.json"
        if not per_well_path.is_file():
            raise FileNotFoundError(f"Métricas por poço ausentes no run {run_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        with per_well_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"Colunas por poço incompletas no run {run_id}")
            records = list(reader)

        expected_wells = int(manifest["dataset"]["n_wells"])
        if len(records) != expected_wells:
            raise ValueError(f"Número de poços divergente no run {run_id}")

        current_population: dict[str, tuple[int, int]] = {}
        f1_values: list[float] = []
        accuracy_values: list[float] = []
        for record in records:
            well_id = record["well_id"]
            if well_id in current_population:
                raise ValueError(f"Poço duplicado em {run_id}: {well_id}")
            n_windows = int(record["n_windows"])
            n_labels = int(record["n_labels"])
            f1_value = float(record["f1_macro_present_labels"])
            accuracy_value = float(record["accuracy"])
            if n_windows <= 0 or n_labels <= 0:
                raise ValueError(f"Suporte inválido para {well_id} no run {run_id}")
            if not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in (f1_value, accuracy_value)
            ):
                raise ValueError(f"Métrica por poço inválida em {run_id}: {well_id}")
            current_population[well_id] = (n_windows, n_labels)
            f1_values.append(f1_value)
            accuracy_values.append(accuracy_value)

        if sum(item[0] for item in current_population.values()) != int(
            manifest["dataset"]["n_windows"]
        ):
            raise ValueError(f"Total de janelas por poço diverge no run {run_id}")
        if population_by_well is None:
            population_by_well = current_population
        elif current_population != population_by_well:
            raise ValueError(f"População por poço incompatível no run {run_id}")

        summaries.append(
            {
                "model": row["model"],
                "display_model": row["display_model"],
                "experiment_id": row["experiment_id"],
                "n_wells": len(records),
                "f1_mean": statistics.fmean(f1_values),
                "f1_median": statistics.median(f1_values),
                "accuracy_mean": statistics.fmean(accuracy_values),
                "accuracy_median": statistics.median(accuracy_values),
                "run_id": run_id,
                "per_well_metrics_sha256": sha256_file(per_well_path),
            }
        )
    return summaries


def _decimal_pt(value: float, *, signed: bool = False) -> str:
    pattern = "+.4f" if signed else ".4f"
    return format(value, pattern).replace(".", ",")


def render_metrics_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Modelo & Exp. & F1-macro & F1-macro $\geq2$ poços & F1 pond. & Bal. acc. & Acurácia \\",
        r"\midrule",
    ]
    previous_model: str | None = None
    for row in rows:
        if previous_model is not None and row["model"] != previous_model:
            lines.append(r"\midrule")
        values = [_decimal_pt(row[column]) for column in METRIC_COLUMNS]
        lines.append(
            f"{row['display_model']} & {row['experiment_id']} & "
            + " & ".join(values)
            + r" \\"
        )
        previous_model = row["model"]
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_contrasts_latex(
    contrasts: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> str:
    model_order = list(dict.fromkeys(row["model"] for row in rows))
    display = {
        row["model"]: "Random Forest" if row["model"] == "rf" else row["display_model"]
        for row in rows
    }
    if len(model_order) != 2:
        raise ValueError("A tabela de contrastes exige exatamente dois modelos")
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        f"Contraste & Fator alterado & {display[model_order[0]]} & {display[model_order[1]]} "
        + r"\\",
        r"\midrule",
    ]
    for contrast in contrasts:
        values = [
            f"${_decimal_pt(contrast['values'][model], signed=True)}$"
            for model in model_order
        ]
        contrast_label = contrast["contrast_id"].replace("-", r" $-$ ")
        lines.append(
            f"{contrast_label} & {contrast['factor_latex']} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_per_well_latex(summaries: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Modelo & Exp. & F1 médio & F1 mediano & Acc. média & Acc. mediana \\",
        r"\midrule",
    ]
    previous_model: str | None = None
    for summary in summaries:
        if previous_model is not None and summary["model"] != previous_model:
            lines.append(r"\midrule")
        values = [
            _decimal_pt(summary[column])
            for column in ("f1_mean", "f1_median", "accuracy_mean", "accuracy_median")
        ]
        lines.append(
            f"{summary['display_model']} & {summary['experiment_id']} & "
            + " & ".join(values)
            + r" \\"
        )
        previous_model = summary["model"]
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def collect_state_support(
    results_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specification = config.get("state_support")
    if not specification:
        raise ValueError("Configuração state_support ausente")
    source_run_id = _source_run_id(specification)
    run_id = _published_run_id(specification)
    run_dir = results_dir / source_run_id
    manifest_path = run_dir / "run_manifest.json"
    features_path = run_dir / "features_causal_real.parquet"
    if not manifest_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(f"Artefatos de suporte ausentes no run {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("run_id") != source_run_id:
        raise ValueError(f"Run de suporte inválido: {source_run_id}")

    features_sha256 = sha256_file(features_path)
    if features_sha256 != manifest["outputs"]["features_sha256"]:
        raise ValueError(f"Hash da tabela de features diverge no run {run_id}")
    frame = pd.read_parquet(
        features_path, columns=["window_label", "instance_id", "well_id"]
    )
    if len(frame) != int(manifest["n_windows"]):
        raise ValueError(f"Número de janelas diverge no run {run_id}")
    if frame["instance_id"].nunique() != int(manifest["n_processed_instances"]):
        raise ValueError(f"Número de instâncias diverge no run {run_id}")
    if frame["well_id"].nunique() != int(manifest["n_wells"]):
        raise ValueError(f"Número de poços diverge no run {run_id}")

    grouped = (
        frame.groupby("window_label", sort=True)
        .agg(
            n_windows=("window_label", "size"),
            n_instances=("instance_id", "nunique"),
            n_wells=("well_id", "nunique"),
        )
        .reset_index()
    )
    rows = [
        {
            "state": int(record.window_label),
            "n_windows": int(record.n_windows),
            "n_instances": int(record.n_instances),
            "n_wells": int(record.n_wells),
        }
        for record in grouped.itertuples(index=False)
    ]
    expected_windows = {
        int(label): int(count)
        for label, count in manifest["windows_by_operational_label"].items()
    }
    observed_windows = {row["state"]: row["n_windows"] for row in rows}
    if observed_windows != expected_windows:
        raise ValueError(f"Suporte por estado diverge do manifesto de {run_id}")
    provenance = {
        "run_id": run_id,
        "project_commit": manifest.get("project_commit"),
        "manifest_sha256": sha256_file(manifest_path),
        "features_sha256": features_sha256,
    }
    return rows, provenance


def render_state_support_latex(rows: list[dict[str, Any]]) -> str:
    def integer_pt(value: int) -> str:
        return f"{value:,}".replace(",", ".")

    split = (len(rows) + 1) // 2
    left, right = rows[:split], rows[split:]
    lines = [
        r"\begin{tabular}{rrrr@{\qquad}rrrr}",
        r"\toprule",
        r"Estado & Janelas & Instâncias & Poços & Estado & Janelas & Instâncias & Poços \\",
        r"\midrule",
    ]
    for index, first in enumerate(left):
        first_cells = [
            str(first["state"]),
            integer_pt(first["n_windows"]),
            str(first["n_instances"]),
            str(first["n_wells"]),
        ]
        if index < len(right):
            second = right[index]
            second_cells = [
                str(second["state"]),
                integer_pt(second["n_windows"]),
                str(second["n_instances"]),
                str(second["n_wells"]),
            ]
        else:
            second_cells = ["", "", "", ""]
        lines.append(" & ".join([*first_cells, *second_cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def collect_shap_sensor_importance(
    results_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specification = config.get("shap_sensor_importance")
    if not specification:
        raise ValueError("Configuração shap_sensor_importance ausente")
    source_run_id = _source_run_id(specification)
    run_id = _published_run_id(specification)
    share_column = str(specification["share_column"])
    run_dir = results_dir / source_run_id
    manifest_path = run_dir / "run_manifest.json"
    importance_path = run_dir / "global_sensor_importance.csv"
    if not manifest_path.is_file() or not importance_path.is_file():
        raise FileNotFoundError(f"Artefatos SHAP ausentes no run {run_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("run_id") != source_run_id
        or manifest.get("experiment_id") != "E3"
        or manifest.get("model") != "random_forest"
    ):
        raise ValueError(f"Identidade do piloto SHAP inválida: {run_id}")

    with importance_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"sensor", share_column}.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Colunas SHAP incompletas no run {run_id}")
        records = list(reader)
    rows = [
        {"sensor": record["sensor"], "importance_share": float(record[share_column])}
        for record in records
    ]
    if len({row["sensor"] for row in rows}) != len(rows):
        raise ValueError(f"Sensor duplicado no run {run_id}")
    if not all(
        math.isfinite(row["importance_share"]) and row["importance_share"] >= 0.0
        for row in rows
    ):
        raise ValueError(f"Importância SHAP inválida no run {run_id}")
    if not math.isclose(sum(row["importance_share"] for row in rows), 1.0, abs_tol=1e-9):
        raise ValueError(f"Importâncias SHAP não somam um no run {run_id}")
    rows.sort(key=lambda row: row["importance_share"], reverse=True)
    provenance = {
        "run_id": run_id,
        "project_commit": manifest.get("project_commit"),
        "manifest_sha256": sha256_file(manifest_path),
        "sensor_importance_sha256": sha256_file(importance_path),
        "n_explanation_windows": int(manifest["explanation_sample"]["n_windows"]),
        "maximum_additivity_residual": float(
            manifest["explanation_sample"]["maximum_additivity_residual"]
        ),
    }
    return rows, provenance


def render_shap_sensor_latex(rows: list[dict[str, Any]]) -> str:
    if len(rows) % 2 != 0:
        raise ValueError("A tabela SHAP em duas colunas exige número par de sensores")
    split = len(rows) // 2
    left, right = rows[:split], rows[split:]
    lines = [
        r"\begin{tabular}{lr@{\qquad}lr}",
        r"\toprule",
        r"Sensor & Importância & Sensor & Importância \\",
        r"\midrule",
    ]
    for first, second in zip(left, right, strict=True):
        first_share = f"{100 * first['importance_share']:.2f}".replace(".", ",")
        second_share = f"{100 * second['importance_share']:.2f}".replace(".", ",")
        lines.append(
            rf"\texttt{{{first['sensor']}}} & {first_share}\% & "
            rf"\texttt{{{second['sensor']}}} & {second_share}\% \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def collect_experiment_matrix(
    project_root: Path, config: dict[str, Any]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    paths = config.get("experiment_matrix_configs")
    if not paths:
        raise ValueError("Configuração experiment_matrix_configs ausente")
    population_labels = {
        ("REAL",): "real",
        ("REAL", "SIMULATED", "DRAWN"): "todas as origens",
    }
    group_labels = {
        "legacy_instance_id": "identidade legada",
        "instance_id": "episódio",
        "well_id": "poço",
    }
    preprocessing_labels = {
        "historical_published_metrics": "histórico publicado",
        "retrospective_baseline_revision": "retrospectivo",
        "causal": "causal",
    }
    rows: list[dict[str, str]] = []
    provenance: list[dict[str, str]] = []
    for relative_path in paths:
        path = project_root / str(relative_path)
        with path.open(encoding="utf-8") as stream:
            experiment = yaml.safe_load(stream)
        sources = tuple(experiment["data"]["sources"])
        group = str(experiment["split"]["group_column"])
        variant = str(experiment["preprocessing"]["variant"])
        if (
            sources not in population_labels
            or group not in group_labels
            or variant not in preprocessing_labels
        ):
            raise ValueError(f"Categoria não mapeada na matriz: {relative_path}")
        rows.append(
            {
                "experiment_id": str(experiment["experiment_id"]),
                "population": population_labels[sources],
                "fold_group": group_labels[group],
                "preprocessing": preprocessing_labels[variant],
            }
        )
        provenance.append(
            {"path": path.relative_to(project_root).as_posix(), "sha256": sha256_file(path)}
        )
    if [row["experiment_id"] for row in rows] != ["R0", "E0", "E1", "E2", "E3"]:
        raise ValueError("Ordem ou identidade inválida na matriz experimental")
    return rows, provenance


def render_experiment_matrix_latex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Código & População & Grupo do fold & Pré-processamento \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["experiment_id"],
                    row["population"],
                    row["fold_group"],
                    row["preprocessing"],
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def collect_r0_reproduction(
    results_dir: Path, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specifications = config.get("r0_reproduction")
    if not specifications:
        raise ValueError("Configuração r0_reproduction ausente")
    metric_labels = {
        "f1_macro": "F1-macro",
        "f1_weighted": "F1 ponderado",
        "accuracy": "Acurácia",
    }
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    dataset_signature: tuple[int, int, int] | None = None
    for specification in specifications:
        source_run_id = _source_run_id(specification)
        run_id = _published_run_id(specification)
        model = str(specification["model"])
        run_dir = results_dir / source_run_id
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "metrics.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "completed"
            or manifest.get("experiment_id") != "R0"
            or manifest.get("model") != model
            or manifest.get("imputation") != "global"
        ):
            raise ValueError(f"Identidade do run R0 inválida: {run_id}")
        dataset = manifest["dataset"]
        signature = (
            int(dataset["n_windows"]),
            int(dataset["n_features"]),
            int(dataset["n_classes"]),
        )
        if dataset_signature is None:
            dataset_signature = signature
        elif signature != dataset_signature:
            raise ValueError(f"População R0 incompatível no run {run_id}")
        comparison = payload["published_comparison"]
        if comparison != manifest["published_comparison"]:
            raise ValueError(f"Comparação publicada diverge do manifesto de {run_id}")
        for metric, label in metric_labels.items():
            item = comparison[metric]
            difference = abs(float(item["published"]) - float(item["reproduced"]))
            if not math.isclose(difference, float(item["absolute_difference"]), abs_tol=5e-5):
                raise ValueError(f"Diferença R0 inválida para {metric} em {run_id}")
            rows.append(
                {
                    "model": model,
                    "display_model": str(specification["display_model"]),
                    "metric": metric,
                    "metric_label": label,
                    "published": float(item["published"]),
                    "reproduced": float(item["reproduced"]),
                    "absolute_difference": float(item["absolute_difference"]),
                    "run_id": run_id,
                }
            )
        provenance.append(
            {
                "run_id": run_id,
                "project_commit": manifest.get("project_commit"),
                "metrics_sha256": sha256_file(metrics_path),
                "manifest_sha256": sha256_file(manifest_path),
                "published_metrics_sha256": manifest.get("published_metrics_sha256"),
            }
        )
    return rows, provenance


def render_r0_reproduction_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Modelo & Métrica & Publicada & Reprodução & Diferença absoluta \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display_model']} & {row['metric_label']} & "
            + " & ".join(
                _decimal_pt(row[column])
                for column in ("published", "reproduced", "absolute_difference")
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_e0_e3_tables(
    *,
    config_path: Path,
    results_dir: Path,
    output_dir: Path,
    manuscript_tables_dir: Path | None = None,
) -> dict[str, Path]:
    config = load_table_config(config_path)
    rows = collect_metric_rows(results_dir, config)
    contrasts = calculate_contrasts(rows, config)
    per_well_summaries = collect_per_well_summaries(results_dir, rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "tab_t004_e0_e3_metrics.csv"
    metrics_tex = output_dir / "tab_t004_e0_e3_metrics.tex"
    contrasts_csv = output_dir / "tab_t005_e0_e3_contrasts.csv"
    contrasts_tex = output_dir / "tab_t005_e0_e3_contrasts.tex"
    per_well_csv = output_dir / "tab_t006_per_well_summary.csv"
    per_well_tex = output_dir / "tab_t006_per_well_summary.tex"
    provenance_path = output_dir / "tab_e0_e3_provenance.json"

    _write_csv(
        metrics_csv,
        [
            "model",
            "display_model",
            "experiment_id",
            *METRIC_COLUMNS,
            "run_id",
            "project_commit",
            "metrics_sha256",
            "manifest_sha256",
        ],
        rows,
    )
    contrast_rows = []
    model_order = list(dict.fromkeys(row["model"] for row in rows))
    for contrast in contrasts:
        contrast_rows.append(
            {
                "contrast_id": contrast["contrast_id"],
                "factor_latex": contrast["factor_latex"],
                **{f"{model}_f1_macro_delta": contrast["values"][model] for model in model_order},
                **{f"{model}_source_runs": contrast["run_ids"][model] for model in model_order},
            }
        )
    _write_csv(contrasts_csv, list(contrast_rows[0]), contrast_rows)
    metrics_tex.write_text(render_metrics_latex(rows), encoding="utf-8", newline="\n")
    contrasts_tex.write_text(
        render_contrasts_latex(contrasts, rows), encoding="utf-8", newline="\n"
    )
    _write_csv(per_well_csv, list(per_well_summaries[0]), per_well_summaries)
    per_well_tex.write_text(
        render_per_well_latex(per_well_summaries), encoding="utf-8", newline="\n"
    )

    project_root = results_dir.parent
    try:
        portable_config_path = config_path.relative_to(project_root).as_posix()
    except ValueError:
        portable_config_path = config_path.as_posix()
    provenance = {
        "schema_version": 1,
        "table_set": config["table_set"],
        "config": {
            "path": portable_config_path,
            "sha256": sha256_file(config_path),
        },
        "inputs": [
            {
                "run_id": row["run_id"],
                "project_commit": row["project_commit"],
                "metrics_sha256": row["metrics_sha256"],
                "manifest_sha256": row["manifest_sha256"],
                "per_well_metrics_sha256": per_well_summaries[index][
                    "per_well_metrics_sha256"
                ],
            }
            for index, row in enumerate(rows)
        ],
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    provenance["outputs"] = {
        path.name: sha256_file(path)
        for path in (
            metrics_csv,
            metrics_tex,
            contrasts_csv,
            contrasts_tex,
            per_well_csv,
            per_well_tex,
        )
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if manuscript_tables_dir is not None:
        manuscript_tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_tex, manuscript_tables_dir / metrics_tex.name)
        shutil.copy2(contrasts_tex, manuscript_tables_dir / contrasts_tex.name)
        shutil.copy2(per_well_tex, manuscript_tables_dir / per_well_tex.name)

    return {
        "metrics_csv": metrics_csv,
        "metrics_tex": metrics_tex,
        "contrasts_csv": contrasts_csv,
        "contrasts_tex": contrasts_tex,
        "per_well_csv": per_well_csv,
        "per_well_tex": per_well_tex,
        "provenance": provenance_path,
    }


def build_support_and_shap_tables(
    *,
    config_path: Path,
    results_dir: Path,
    output_dir: Path,
    manuscript_tables_dir: Path | None = None,
) -> dict[str, Path]:
    config = load_table_config(config_path)
    support_rows, support_provenance = collect_state_support(results_dir, config)
    shap_rows, shap_provenance = collect_shap_sensor_importance(results_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_csv = output_dir / "tab_t003_state_support.csv"
    support_tex = output_dir / "tab_t003_state_support.tex"
    shap_csv = output_dir / "tab_t007_shap_sensor_importance.csv"
    shap_tex = output_dir / "tab_t007_shap_sensor_importance.tex"
    provenance_path = output_dir / "tab_support_shap_provenance.json"

    _write_csv(support_csv, list(support_rows[0]), support_rows)
    support_tex.write_text(
        render_state_support_latex(support_rows), encoding="utf-8", newline="\n"
    )
    _write_csv(shap_csv, list(shap_rows[0]), shap_rows)
    shap_tex.write_text(
        render_shap_sensor_latex(shap_rows), encoding="utf-8", newline="\n"
    )

    project_root = results_dir.parent
    try:
        portable_config_path = config_path.relative_to(project_root).as_posix()
    except ValueError:
        portable_config_path = config_path.as_posix()
    provenance = {
        "schema_version": 1,
        "table_set": "state_support_and_shap_pilot",
        "config": {
            "path": portable_config_path,
            "sha256": sha256_file(config_path),
        },
        "inputs": {
            "state_support": support_provenance,
            "shap_sensor_importance": shap_provenance,
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (support_csv, support_tex, shap_csv, shap_tex)
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if manuscript_tables_dir is not None:
        manuscript_tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(support_tex, manuscript_tables_dir / support_tex.name)
        shutil.copy2(shap_tex, manuscript_tables_dir / shap_tex.name)

    return {
        "support_csv": support_csv,
        "support_tex": support_tex,
        "shap_csv": shap_csv,
        "shap_tex": shap_tex,
        "provenance": provenance_path,
    }


def build_method_and_r0_tables(
    *,
    config_path: Path,
    results_dir: Path,
    output_dir: Path,
    manuscript_tables_dir: Path | None = None,
) -> dict[str, Path]:
    config = load_table_config(config_path)
    project_root = results_dir.parent
    matrix_rows, matrix_inputs = collect_experiment_matrix(project_root, config)
    r0_rows, r0_inputs = collect_r0_reproduction(results_dir, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = output_dir / "tab_t001_experiment_matrix.csv"
    matrix_tex = output_dir / "tab_t001_experiment_matrix.tex"
    r0_csv = output_dir / "tab_t002_r0_reproduction.csv"
    r0_tex = output_dir / "tab_t002_r0_reproduction.tex"
    provenance_path = output_dir / "tab_method_r0_provenance.json"

    _write_csv(matrix_csv, list(matrix_rows[0]), matrix_rows)
    matrix_tex.write_text(
        render_experiment_matrix_latex(matrix_rows), encoding="utf-8", newline="\n"
    )
    _write_csv(r0_csv, list(r0_rows[0]), r0_rows)
    r0_tex.write_text(
        render_r0_reproduction_latex(r0_rows), encoding="utf-8", newline="\n"
    )
    provenance = {
        "schema_version": 1,
        "table_set": "method_matrix_and_r0_reproduction",
        "config": {
            "path": config_path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "inputs": {"experiment_configs": matrix_inputs, "r0_runs": r0_inputs},
        "outputs": {
            path.name: sha256_file(path)
            for path in (matrix_csv, matrix_tex, r0_csv, r0_tex)
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if manuscript_tables_dir is not None:
        manuscript_tables_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matrix_tex, manuscript_tables_dir / matrix_tex.name)
        shutil.copy2(r0_tex, manuscript_tables_dir / r0_tex.name)
    return {
        "matrix_csv": matrix_csv,
        "matrix_tex": matrix_tex,
        "r0_csv": r0_csv,
        "r0_tex": r0_tex,
        "provenance": provenance_path,
    }
