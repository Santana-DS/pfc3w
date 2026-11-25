"""Construção e auditoria de separações por unidade física."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def estimate_window_count(num_rows: int, window_size: int = 300, step_size: int = 150) -> int:
    """Estima janelas completas antes dos filtros de qualidade e rotulagem."""
    if num_rows < 0:
        raise ValueError("num_rows não pode ser negativo")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size e step_size devem ser positivos")
    if num_rows < window_size:
        return 0
    return (num_rows - window_size) // step_size + 1


def weighted_group_kfold_assignment(
    frame: pd.DataFrame,
    *,
    group_column: str,
    weight_column: str,
    n_splits: int,
) -> pd.DataFrame:
    """Atribui grupos a folds equilibrando um peso inteiro por grupo.

    A expansão dos pesos reproduz o comportamento de ``GroupKFold`` caso cada
    unidade de peso fosse uma amostra. No inventário estrutural, o peso é o
    número estimado de janelas de cada arquivo.
    """
    required = {group_column, weight_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Colunas ausentes: {sorted(missing)}")
    if frame[group_column].isna().any():
        raise ValueError(f"{group_column} contém valores ausentes")

    weights = frame[weight_column].to_numpy()
    if not np.issubdtype(weights.dtype, np.integer):
        if not np.all(np.equal(weights, np.floor(weights))):
            raise ValueError(f"{weight_column} precisa conter pesos inteiros")
        weights = weights.astype(int)
    if np.any(weights < 0):
        raise ValueError(f"{weight_column} contém pesos negativos")

    groups = frame[group_column].astype(str).to_numpy()
    expanded_groups = np.repeat(groups, weights)
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Há {len(unique_groups)} grupos, menos que os {n_splits} folds solicitados"
        )
    if len(expanded_groups) == 0:
        raise ValueError("Nenhuma amostra com peso positivo")

    fold_by_group: dict[str, int] = {}
    splitter = GroupKFold(n_splits=n_splits)
    placeholder = np.zeros(len(expanded_groups), dtype=np.uint8)
    for fold, (_, test_index) in enumerate(splitter.split(placeholder, groups=expanded_groups)):
        for group in np.unique(expanded_groups[test_index]):
            fold_by_group[str(group)] = fold

    group_weights = (
        frame.assign(**{group_column: groups})
        .groupby(group_column, as_index=False)[weight_column]
        .sum()
    )
    group_weights["fold"] = group_weights[group_column].map(fold_by_group).astype(int)
    return group_weights.sort_values(["fold", group_column]).reset_index(drop=True)


def class_coverage_by_fold(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    group_column: str,
    class_column: str,
    weight_column: str,
) -> pd.DataFrame:
    """Conta suporte de treino e teste de cada classe em cada fold."""
    assignment_map = assignments.set_index(group_column)["fold"]
    if assignment_map.index.has_duplicates:
        raise ValueError("Um grupo recebeu mais de um fold")

    data = frame.copy()
    data["fold"] = data[group_column].map(assignment_map)
    if data["fold"].isna().any():
        raise ValueError("Há grupos sem fold atribuído")

    rows = []
    classes = sorted(data[class_column].unique())
    for fold in sorted(assignments["fold"].unique()):
        test = data[data["fold"] == fold]
        train = data[data["fold"] != fold]
        for class_id in classes:
            train_class = train[train[class_column] == class_id]
            test_class = test[test[class_column] == class_id]
            rows.append(
                {
                    "fold": int(fold),
                    "class_id": int(class_id),
                    "train_instances": int(len(train_class)),
                    "test_instances": int(len(test_class)),
                    "train_weight": int(train_class[weight_column].sum()),
                    "test_weight": int(test_class[weight_column].sum()),
                    "train_present": not train_class.empty,
                    "test_present": not test_class.empty,
                }
            )
    return pd.DataFrame(rows)


def window_coverage_by_fold(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Resume suporte de janelas e grupos para cada classe em cada fold."""
    required_frame = {group_column, label_column}
    missing_frame = required_frame - set(frame.columns)
    if missing_frame:
        raise KeyError(f"Colunas ausentes nas janelas: {sorted(missing_frame)}")
    required_assignments = {group_column, "fold"}
    missing_assignments = required_assignments - set(assignments.columns)
    if missing_assignments:
        raise KeyError(f"Colunas ausentes nas atribuições: {sorted(missing_assignments)}")
    if assignments[group_column].duplicated().any():
        raise ValueError("Um grupo recebeu mais de uma atribuição")

    assignment_map = assignments.set_index(group_column)["fold"]
    data = frame[[group_column, label_column]].copy()
    data["fold"] = data[group_column].map(assignment_map)
    if data["fold"].isna().any():
        raise ValueError("Há janelas pertencentes a grupos sem fold")

    rows: list[dict[str, int | bool]] = []
    labels = sorted(data[label_column].unique())
    for fold in sorted(assignments["fold"].unique()):
        test = data[data["fold"] == fold]
        train = data[data["fold"] != fold]
        for label in labels:
            train_label = train[train[label_column] == label]
            test_label = test[test[label_column] == label]
            rows.append(
                {
                    "fold": int(fold),
                    "label": int(label),
                    "train_windows": int(len(train_label)),
                    "test_windows": int(len(test_label)),
                    "train_groups": int(train_label[group_column].nunique()),
                    "test_groups": int(test_label[group_column].nunique()),
                    "train_present": not train_label.empty,
                    "test_present": not test_label.empty,
                }
            )
    return pd.DataFrame(rows)
