import numpy as np
import pandas as pd

from pfc3w.causal import (
    causal_gaussian_filter,
    expanding_zscore,
    extract_causal_features_from_instance,
    impute_from_training_fold,
)


def test_expanding_zscore_is_prefix_invariant_and_preserves_missing_values() -> None:
    values = np.array([2.0, 3.0, np.nan, 6.0, 4.0, 9.0])
    full = expanding_zscore(values)

    np.testing.assert_allclose(full[:4], expanding_zscore(values[:4]), equal_nan=True)
    assert full[0] == 0.0
    assert np.isnan(full[2])


def test_causal_gaussian_filter_is_prefix_invariant() -> None:
    values = np.array([0.0, 1.0, 4.0, np.nan, 2.0, 8.0])
    full = causal_gaussian_filter(values, sigma=2.0)

    np.testing.assert_allclose(
        full[:4],
        causal_gaussian_filter(values[:4], sigma=2.0),
        equal_nan=True,
    )
    assert np.isnan(full[3])


def test_future_changes_do_not_alter_completed_feature_windows() -> None:
    random = np.random.default_rng(42)
    n_rows = 750
    frame = pd.DataFrame(
        {
            "P-TPT": random.normal(size=n_rows).cumsum(),
            "QGL": random.normal(size=n_rows),
            "class": np.where(np.arange(n_rows) < 500, 0, 1),
            "instance_id": "1/example",
            "fault_class": 1,
            "source_type": "REAL",
        },
        index=pd.date_range("2020-01-01", periods=n_rows, freq="s"),
    )
    changed_future = frame.copy()
    changed_future.loc[changed_future.index[600]:, "P-TPT"] += 1_000_000.0
    changed_future.loc[changed_future.index[600]:, "QGL"] *= -10_000.0

    common = {
        "sensors": ["P-TPT", "QGL"],
        "window_size": 300,
        "step_size": 150,
        "gaussian_sigma": 2.0,
    }
    original_features = extract_causal_features_from_instance(frame, **common)
    changed_features = extract_causal_features_from_instance(changed_future, **common)

    # As janelas iniciadas em 0, 150 e 300 terminam antes da intervenção em t=600.
    pd.testing.assert_frame_equal(
        original_features.iloc[:3],
        changed_features.iloc[:3],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.allclose(
        original_features.iloc[3].filter(like="P-TPT").to_numpy(dtype=float),
        changed_features.iloc[3].filter(like="P-TPT").to_numpy(dtype=float),
    )


def test_imputation_uses_only_training_rows_and_zero_for_empty_column() -> None:
    raw = np.array(
        [
            [np.nan, np.nan, 1.0],
            [1.0, np.nan, np.nan],
            [3.0, np.nan, 5.0],
            [1000.0, 7.0, np.nan],
        ]
    )
    imputed = impute_from_training_fold(raw, np.array([True, True, True, False]))

    np.testing.assert_allclose(imputed[0], [2.0, 0.0, 1.0])
    np.testing.assert_allclose(imputed[1], [1.0, 0.0, 3.0])
    np.testing.assert_allclose(imputed[3], [1000.0, 7.0, 3.0])
