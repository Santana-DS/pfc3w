"""Pré-processamento causal e extração de features temporais do 3W."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from pfc3w.features import FEATURE_STATISTICS

CONSTANT_THRESHOLD = 1e-10


def expanding_zscore(
    values: np.ndarray,
    *,
    constant_threshold: float = CONSTANT_THRESHOLD,
) -> np.ndarray:
    """Normaliza cada valor com média e desvio do prefixo disponível.

    As estatísticas no instante ``t`` incluem somente observações de ``0`` a
    ``t``. O desvio é populacional (``ddof=0``), como no baseline. Uma
    observação é mapeada para zero enquanto há menos de duas observações
    válidas ou o desvio é numericamente nulo. Valores ausentes permanecem NaN.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("expanding_zscore aceita somente uma série unidimensional")
    if constant_threshold < 0:
        raise ValueError("constant_threshold não pode ser negativo")

    series = pd.Series(array, copy=False)
    expanding = series.expanding(min_periods=1)
    means = expanding.mean().to_numpy()
    standard_deviations = expanding.std(ddof=0).to_numpy()
    counts = expanding.count().to_numpy()

    observed = ~np.isnan(array)
    stable = observed & (counts >= 2) & (standard_deviations >= constant_threshold)
    result = np.full(array.shape, np.nan, dtype=float)
    result[observed & ~stable] = 0.0
    np.divide(array - means, standard_deviations, out=result, where=stable)
    return result


def causal_gaussian_filter(
    values: np.ndarray,
    *,
    sigma: float,
    truncate: float = 4.0,
) -> np.ndarray:
    """Aplica média gaussiana unilateral sem consultar amostras futuras.

    O peso do atraso ``k`` é ``exp(-0.5 * (k / sigma) ** 2)``. Na borda
    inicial, os pesos disponíveis são renormalizados. NaNs não contribuem para
    numerador ou denominador e permanecem NaN no instante em que ocorrem.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("causal_gaussian_filter aceita somente uma série unidimensional")
    if sigma <= 0:
        raise ValueError("sigma precisa ser positivo")
    if truncate <= 0:
        raise ValueError("truncate precisa ser positivo")
    if array.size == 0:
        return array.copy()

    maximum_lag = int(truncate * sigma + 0.5)
    lags = np.arange(maximum_lag + 1, dtype=float)
    weights = np.exp(-0.5 * (lags / sigma) ** 2)

    observed = ~np.isnan(array)
    filled = np.where(observed, array, 0.0)
    numerator = np.convolve(filled, weights, mode="full")[: array.size]
    denominator = np.convolve(observed.astype(float), weights, mode="full")[: array.size]

    result = np.full(array.shape, np.nan, dtype=float)
    valid = observed & (denominator > 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def transform_instance_causally(
    frame: pd.DataFrame,
    *,
    sensors: Sequence[str],
    gaussian_sigma: float,
    gaussian_truncate: float = 4.0,
) -> pd.DataFrame:
    """Normaliza e suaviza uma instância usando apenas prefixos temporais."""
    transformed = frame.copy()
    for sensor in sensors:
        if sensor not in transformed.columns:
            transformed[sensor] = np.nan
        normalized = expanding_zscore(transformed[sensor].to_numpy(dtype=float))
        transformed[sensor] = causal_gaussian_filter(
            normalized,
            sigma=gaussian_sigma,
            truncate=gaussian_truncate,
        )
    return transformed


def impute_from_training_fold(
    raw_features: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Imputa toda a matriz com medianas calculadas somente no treino."""
    features = np.asarray(raw_features, dtype=float)
    mask = np.asarray(train_mask, dtype=bool)
    if features.ndim != 2:
        raise ValueError("raw_features precisa ser uma matriz bidimensional")
    if mask.ndim != 1 or len(mask) != len(features):
        raise ValueError("train_mask precisa ter uma entrada por linha")
    if not mask.any():
        raise ValueError("train_mask não pode selecionar um treino vazio")

    training = pd.DataFrame(features[mask])
    medians = training.median(axis=0, skipna=True).fillna(0.0).to_numpy(dtype=float)
    return np.where(np.isnan(features), medians, features)


def extract_window_statistics(window: np.ndarray, sensor_name: str) -> dict[str, float]:
    """Extrai as mesmas onze estatísticas do baseline de uma janela."""
    valid = np.asarray(window, dtype=float)
    valid = valid[~np.isnan(valid)]
    if len(valid) < 10:
        return {f"{sensor_name}_{name}": np.nan for name in FEATURE_STATISTICS}

    mean_value = valid.mean()
    standard_deviation = valid.std()
    first_difference = np.diff(valid)
    second_difference = np.diff(first_difference)
    q75, q25 = np.percentile(valid, [75, 25])

    if standard_deviation < CONSTANT_THRESHOLD:
        skewness = 0.0
        excess_kurtosis = 0.0
        maximum_zscore = 0.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = float(skew(valid))
            excess_kurtosis = float(kurtosis(valid))
        maximum_zscore = float(
            (np.abs(valid - mean_value) / standard_deviation).max()
        )

    return {
        f"{sensor_name}_mean": float(mean_value),
        f"{sensor_name}_std": float(standard_deviation),
        f"{sensor_name}_min": float(valid.min()),
        f"{sensor_name}_max": float(valid.max()),
        f"{sensor_name}_skewness": skewness,
        f"{sensor_name}_kurtosis": excess_kurtosis,
        f"{sensor_name}_median": float(np.median(valid)),
        f"{sensor_name}_iqr": float(q75 - q25),
        f"{sensor_name}_diff1_std": (
            float(first_difference.std()) if len(first_difference) > 1 else np.nan
        ),
        f"{sensor_name}_diff2_std": (
            float(second_difference.std()) if len(second_difference) > 1 else np.nan
        ),
        f"{sensor_name}_max_zscore": maximum_zscore,
    }


def extract_causal_features_from_instance(
    frame: pd.DataFrame,
    *,
    sensors: Sequence[str],
    window_size: int,
    step_size: int,
    gaussian_sigma: float,
    gaussian_truncate: float = 4.0,
) -> pd.DataFrame:
    """Extrai features de janelas encerradas no passado ou no presente.

    A decisão associada a uma janela iniciada em ``s`` ocorre depois de
    ``s + window_size - 1``. A classe operacional é usada somente como alvo e
    corresponde à moda dos rótulos disponíveis nessa mesma janela.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size e step_size precisam ser positivos")
    required = {"instance_id", "fault_class", "source_type", "class"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    transformed = transform_instance_causally(
        frame,
        sensors=sensors,
        gaussian_sigma=gaussian_sigma,
        gaussian_truncate=gaussian_truncate,
    )
    rows: list[dict[str, object]] = []
    for start in range(0, len(transformed) - window_size + 1, step_size):
        window = transformed.iloc[start : start + window_size]
        valid_states = window["class"].dropna()
        if valid_states.empty:
            continue
        row: dict[str, object] = {
            "instance_id": transformed["instance_id"].iloc[0],
            "fault_class": transformed["fault_class"].iloc[0],
            "source_type": transformed["source_type"].iloc[0],
            "window_start": start,
            "window_label": int(valid_states.mode().iloc[0]),
        }
        for sensor in sensors:
            row.update(extract_window_statistics(window[sensor].to_numpy(), sensor))
        rows.append(row)
    return pd.DataFrame(rows)
