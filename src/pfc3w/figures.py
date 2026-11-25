"""Geração rastreável de figuras acadêmicas selecionadas para o relatório."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shap_feature_importance(
    run_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    importance_path = run_dir / "global_feature_importance.csv"
    if not manifest_path.is_file() or not importance_path.is_file():
        raise FileNotFoundError(f"Artefatos da figura SHAP ausentes em {run_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("experiment_id") != "E3"
        or manifest.get("model") != "random_forest"
        or manifest.get("fold") != 0
    ):
        raise ValueError("A F001 exige o piloto concluído de Random Forest no fold 0 de E3")

    frame = pd.read_csv(importance_path)
    required = {"feature", "sensor", "statistic", "predicted_output_share"}
    if not required.issubset(frame.columns):
        raise ValueError("Colunas obrigatórias ausentes na importância SHAP")
    if frame["feature"].duplicated().any():
        raise ValueError("Features duplicadas na importância SHAP")
    shares = frame["predicted_output_share"].astype(float)
    if not all(math.isfinite(value) and value >= 0.0 for value in shares):
        raise ValueError("Fração SHAP inválida")
    if not math.isclose(float(shares.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("As frações SHAP por feature não somam um")
    frame = frame.assign(predicted_output_share=shares).sort_values(
        "predicted_output_share", ascending=False
    )
    provenance = {
        "run_id": manifest["run_id"],
        "project_commit": manifest.get("project_commit"),
        "manifest_sha256": _sha256_file(manifest_path),
        "feature_importance_sha256": _sha256_file(importance_path),
        "n_explanation_windows": int(manifest["explanation_sample"]["n_windows"]),
        "sample_selection": manifest["explanation_sample"]["selection"],
        "maximum_additivity_residual": float(
            manifest["explanation_sample"]["maximum_additivity_residual"]
        ),
    }
    return frame, manifest, provenance


def render_f001_shap_global(
    frame: pd.DataFrame,
    *,
    pdf_path: Path,
    png_path: Path,
    top_n: int = 15,
) -> None:
    if top_n <= 0 or top_n > len(frame):
        raise ValueError("top_n incompatível com a tabela de importância")
    selected = frame.head(top_n).iloc[::-1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    bars = axis.barh(
        selected["feature"],
        selected["predicted_output_share"],
        color="#0072B2",
        edgecolor="none",
        height=0.72,
    )
    maximum = float(selected["predicted_output_share"].max())
    axis.set_xlim(0.0, maximum * 1.22)
    axis.set_xlabel("Fração da importância SHAP absoluta para a classe prevista")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.xaxis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for bar, value in zip(bars, selected["predicted_output_share"], strict=True):
        axis.text(
            float(value) + maximum * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{100 * float(value):.2f}%".replace(".", ","),
            va="center",
            ha="left",
            fontsize=7.5,
            color="#222222",
        )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "pfc3w", "CreationDate": None},
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_f001_shap_global(
    *,
    run_dir: Path,
    output_dir: Path,
    manuscript_figures_dir: Path | None = None,
    top_n: int = 15,
    scientific_run_id: str | None = None,
) -> dict[str, Path]:
    frame, _manifest, provenance = load_shap_feature_importance(run_dir)
    if scientific_run_id is not None:
        provenance["run_id"] = scientific_run_id
    pdf_path = output_dir / "fig_f001_shap_global_e3.pdf"
    png_path = output_dir / "fig_f001_shap_global_e3.png"
    provenance_path = output_dir / "fig_f001_shap_global_e3_provenance.json"
    render_f001_shap_global(
        frame, pdf_path=pdf_path, png_path=png_path, top_n=top_n
    )
    provenance.update(
        {
            "schema_version": 1,
            "figure_id": "F001",
            "top_n": top_n,
            "outputs": {
                pdf_path.name: _sha256_file(pdf_path),
                png_path.name: _sha256_file(png_path),
            },
        }
    )
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if manuscript_figures_dir is not None:
        manuscript_figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, manuscript_figures_dir / pdf_path.name)
    return {"pdf": pdf_path, "png": png_path, "provenance": provenance_path}


def _write_figure_provenance(
    *,
    figure_id: str,
    input_paths: list[Path],
    output_paths: list[Path],
    provenance_path: Path,
    details: dict[str, Any],
    input_labels: list[str] | None = None,
) -> None:
    labels = input_labels or [path.name for path in input_paths]
    if len(labels) != len(input_paths) or len(set(labels)) != len(labels):
        raise ValueError("Rótulos de entrada precisam ser únicos e completos")
    payload = {
        "schema_version": 1,
        "figure_id": figure_id,
        "inputs": {
            label: _sha256_file(path)
            for label, path in zip(labels, input_paths, strict=True)
        },
        "outputs": {
            path.name: _sha256_file(path) for path in output_paths
        },
        **details,
    }
    provenance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _save_figure(figure: plt.Figure, *, pdf_path: Path, png_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Creator": "pfc3w", "CreationDate": None},
    )
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_f002_e0_e3_metrics(
    *, metrics_csv: Path, output_dir: Path, manuscript_figures_dir: Path | None = None
) -> dict[str, Path]:
    frame = pd.read_csv(metrics_csv)
    required = {
        "model",
        "display_model",
        "experiment_id",
        "f1_macro_all_labels",
        "run_id",
        "metrics_sha256",
        "manifest_sha256",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Colunas obrigatórias ausentes na tabela E0--E3")
    expected_pairs = {
        (model, experiment)
        for model in ("rf", "xgboost")
        for experiment in ("E0", "E1", "E2", "E3")
    }
    observed_pairs = set(zip(frame["model"], frame["experiment_id"], strict=True))
    if observed_pairs != expected_pairs:
        raise ValueError("A F002 exige os oito pares modelo--experimento de E0--E3")

    order = ["E0", "E1", "E2", "E3"]
    labels = [
        "E0\nEpisódio\nRetrospectivo",
        "E1\nPoço\nRetrospectivo",
        "E2\nEpisódio\nCausal",
        "E3\nPoço\nCausal",
    ]
    colors = {"rf": "#0072B2", "xgboost": "#D55E00"}
    x = np.arange(len(order), dtype=float)
    width = 0.34
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    for index, model in enumerate(("rf", "xgboost")):
        subset = frame[frame["model"] == model].set_index("experiment_id").loc[order]
        values = subset["f1_macro_all_labels"].astype(float).to_numpy()
        offset = (index - 0.5) * width
        bars = axis.bar(
            x + offset,
            values,
            width=width,
            color=colors[model],
            edgecolor="white",
            linewidth=0.5,
            label="Random Forest" if model == "rf" else "XGBoost",
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=7.5)
    axis.set_ylabel("F1-macro (17 estados)")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 0.55)
    axis.yaxis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=2, loc="upper center")

    pdf_path = output_dir / "fig_f002_e0_e3_f1_macro.pdf"
    png_path = output_dir / "fig_f002_e0_e3_f1_macro.png"
    provenance_path = output_dir / "fig_f002_e0_e3_f1_macro_provenance.json"
    _save_figure(figure, pdf_path=pdf_path, png_path=png_path)
    _write_figure_provenance(
        figure_id="F002",
        input_paths=[metrics_csv],
        output_paths=[pdf_path, png_path],
        provenance_path=provenance_path,
        details={
            "metric": "f1_macro_all_labels",
            "run_ids": frame["run_id"].tolist(),
            "interpretation": "controlled E0--E3 comparison; R0 excluded",
        },
        input_labels=[f"reports/tables/{metrics_csv.name}"],
    )
    if manuscript_figures_dir is not None:
        manuscript_figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, manuscript_figures_dir / pdf_path.name)
    return {"pdf": pdf_path, "png": png_path, "provenance": provenance_path}


def build_f003_state_support(*, support_csv: Path, output_dir: Path) -> dict[str, Path]:
    frame = pd.read_csv(support_csv)
    required = {"state", "n_windows", "n_instances", "n_wells"}
    if not required.issubset(frame.columns) or frame["state"].duplicated().any():
        raise ValueError("Tabela de suporte dos estados inválida")
    if (frame[["n_windows", "n_instances", "n_wells"]] <= 0).any().any():
        raise ValueError("Suporte não positivo na F003")
    frame = frame.sort_values("state", key=lambda values: values.astype(int))
    y = np.arange(len(frame))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    figure, (axis_windows, axis_wells) = plt.subplots(
        1, 2, figsize=(7.0, 5.2), sharey=True, constrained_layout=True
    )
    axis_windows.barh(y, frame["n_windows"], color="#0072B2", height=0.7)
    axis_windows.set_xscale("log")
    axis_windows.set_xlabel("Janelas (escala log)")
    axis_windows.set_ylabel("Estado operacional")
    axis_windows.set_yticks(y, frame["state"].astype(str))
    axis_windows.invert_yaxis()
    axis_windows.xaxis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis_wells.barh(y, frame["n_wells"], color="#009E73", height=0.7)
    axis_wells.set_xlabel("Poços com suporte")
    axis_wells.set_xlim(0, max(40, int(frame["n_wells"].max()) + 2))
    axis_wells.xaxis.grid(True, color="#D9D9D9", linewidth=0.6)
    support_pairs = zip(frame["n_windows"], frame["n_wells"], strict=True)
    for index, (windows, wells) in enumerate(support_pairs):
        window_label = f"{int(windows):,}".replace(",", ".")
        axis_windows.text(
            float(windows) * 1.08, index, window_label, va="center", fontsize=7
        )
        axis_wells.text(float(wells) + 0.4, index, str(int(wells)), va="center", fontsize=7)
    for axis in (axis_windows, axis_wells):
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    pdf_path = output_dir / "fig_f003_state_support.pdf"
    png_path = output_dir / "fig_f003_state_support.png"
    provenance_path = output_dir / "fig_f003_state_support_provenance.json"
    _save_figure(figure, pdf_path=pdf_path, png_path=png_path)
    _write_figure_provenance(
        figure_id="F003",
        input_paths=[support_csv],
        output_paths=[pdf_path, png_path],
        provenance_path=provenance_path,
        details={
            "population": {
                "n_windows": int(frame["n_windows"].sum()),
                "n_states": len(frame),
            }
        },
        input_labels=[f"reports/tables/{support_csv.name}"],
    )
    return {"pdf": pdf_path, "png": png_path, "provenance": provenance_path}


def build_f004_per_well_distribution(
    *,
    run_specifications: list[tuple[str, str, str, str]],
    results_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    frames: list[pd.DataFrame] = []
    input_paths: list[Path] = []
    for experiment_id, model, source_run_id, scientific_run_id in run_specifications:
        path = results_dir / source_run_id / "per_well_metrics.csv"
        frame = pd.read_csv(path)
        required = {"well_id", "f1_macro_present_labels"}
        if not required.issubset(frame.columns) or len(frame) != 35:
            raise ValueError(f"Métricas por poço inválidas em {source_run_id}")
        frame = frame.assign(
            experiment_id=experiment_id, model=model, run_id=scientific_run_id
        )
        frames.append(frame)
        input_paths.append(path)
    data = pd.concat(frames, ignore_index=True)
    expected = {("E1", "rf"), ("E1", "xgboost"), ("E3", "rf"), ("E3", "xgboost")}
    if set(zip(data["experiment_id"], data["model"], strict=True)) != expected:
        raise ValueError("A F004 exige RF e XGBoost em E1 e E3")

    groups = [("E1", "rf"), ("E1", "xgboost"), ("E3", "rf"), ("E3", "xgboost")]
    positions = [0.0, 1.0, 2.6, 3.6]
    labels = ["E1\nRF", "E1\nXGBoost", "E3\nRF", "E3\nXGBoost"]
    colors = ["#0072B2", "#D55E00", "#56B4E9", "#E69F00"]
    values = [
        data[(data["experiment_id"] == experiment) & (data["model"] == model)][
            "f1_macro_present_labels"
        ].astype(float).to_numpy()
        for experiment, model in groups
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.3},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    generator = np.random.default_rng(42)
    for position, group_values, color in zip(positions, values, colors, strict=True):
        jitter = generator.uniform(-0.16, 0.16, size=len(group_values))
        axis.scatter(
            position + jitter,
            group_values,
            s=13,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.8,
            zorder=3,
        )
    axis.axvline(1.8, color="#B0B0B0", linewidth=0.8)
    axis.text(0.5, 1.025, "Retrospectivo", transform=axis.get_xaxis_transform(), ha="center")
    axis.text(3.1, 1.025, "Causal", transform=axis.get_xaxis_transform(), ha="center")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("F1-macro por poço (rótulos presentes)")
    axis.set_ylim(-0.02, 1.03)
    axis.yaxis.grid(True, color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    pdf_path = output_dir / "fig_f004_per_well_f1_distribution.pdf"
    png_path = output_dir / "fig_f004_per_well_f1_distribution.png"
    provenance_path = output_dir / "fig_f004_per_well_f1_distribution_provenance.json"
    _save_figure(figure, pdf_path=pdf_path, png_path=png_path)
    _write_figure_provenance(
        figure_id="F004",
        input_paths=input_paths,
        output_paths=[pdf_path, png_path],
        provenance_path=provenance_path,
        details={
            "run_ids": [item[3] for item in run_specifications],
            "n_wells_per_group": 35,
        },
        input_labels=[
            f"results/{item[3]}/per_well_metrics.csv" for item in run_specifications
        ],
    )
    return {"pdf": pdf_path, "png": png_path, "provenance": provenance_path}


def build_f005_e3_confusion_matrices(
    *,
    run_specifications: list[tuple[str, str, str]],
    results_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 101, 102, 105, 106, 107, 108, 109]
    matrices: list[np.ndarray] = []
    input_paths: list[Path] = []
    for _display_model, source_run_id, _scientific_run_id in run_specifications:
        path = results_dir / source_run_id / "oof_predictions.parquet"
        frame = pd.read_parquet(path, columns=["y_true", "y_pred"])
        if len(frame) != 159779:
            raise ValueError(f"População OOF inválida em {source_run_id}")
        matrices.append(
            confusion_matrix(
                frame["y_true"], frame["y_pred"], labels=labels, normalize="true"
            )
        )
        input_paths.append(path)
    if len(matrices) != 2:
        raise ValueError("A F005 exige duas matrizes E3")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
    image_handle = None
    figure_items = zip(axes, matrices, run_specifications, strict=True)
    for axis, matrix, (display_model, _source_run_id, _scientific_run_id) in figure_items:
        image_handle = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
        axis.set_title(display_model)
        axis.set_xticks(range(len(labels)), [str(label) for label in labels], rotation=90)
        axis.set_yticks(range(len(labels)), [str(label) for label in labels])
        axis.set_xlabel("Estado previsto")
        axis.set_ylabel("Estado verdadeiro")
    if image_handle is not None:
        colorbar = figure.colorbar(image_handle, ax=axes, shrink=0.86, pad=0.02)
        colorbar.set_label("Proporção por estado verdadeiro")

    pdf_path = output_dir / "fig_f005_e3_confusion_matrices.pdf"
    png_path = output_dir / "fig_f005_e3_confusion_matrices.png"
    provenance_path = output_dir / "fig_f005_e3_confusion_matrices_provenance.json"
    _save_figure(figure, pdf_path=pdf_path, png_path=png_path)
    _write_figure_provenance(
        figure_id="F005",
        input_paths=input_paths,
        output_paths=[pdf_path, png_path],
        provenance_path=provenance_path,
        details={
            "normalization": "rows (true-label support)",
            "labels": labels,
            "run_ids": [item[2] for item in run_specifications],
        },
        input_labels=[
            f"results/{item[2]}/oof_predictions.parquet" for item in run_specifications
        ],
    )
    return {"pdf": pdf_path, "png": png_path, "provenance": provenance_path}
