from tempfile import gettempdir
from os.path import join
import warnings

import numpy as np
from numpy.testing import assert_allclose

import pytest

import tslearn.utils

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


EXAMPLE_FILE = join(gettempdir(), "tslearn_pytest_file.txt")


def test_save_load_random():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    dataset = rng.randn(n, sz, d)
    tslearn.utils.save_timeseries_txt(EXAMPLE_FILE, dataset)
    reloaded_dataset = tslearn.utils.load_timeseries_txt(EXAMPLE_FILE)
    assert_allclose(dataset, reloaded_dataset)


def test_save_load_known():
    dataset = tslearn.utils.to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3]])
    tslearn.utils.save_timeseries_txt(EXAMPLE_FILE, dataset)
    reloaded_dataset = tslearn.utils.load_timeseries_txt(EXAMPLE_FILE)
    for ts0, ts1 in zip(dataset, reloaded_dataset):
        assert_allclose(ts0[:tslearn.utils.ts_size(ts0)],
                        ts1[:tslearn.utils.ts_size(ts1)])


def test_conversions():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    tslearn_dataset = rng.randn(n, sz, d)

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_pyts_dataset(
            tslearn.utils.to_pyts_dataset(tslearn_dataset)
        )
    )

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_seglearn_dataset(
            tslearn.utils.to_seglearn_dataset(tslearn_dataset)
        )
    )

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_stumpy_dataset(
            tslearn.utils.to_stumpy_dataset(tslearn_dataset)
        )
    )


def test_conversions_with_pandas():
    pytest.importorskip('pandas')
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    tslearn_dataset = rng.randn(n, sz, d)

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_sktime_dataset(
            tslearn.utils.to_sktime_dataset(tslearn_dataset)
        )
    )

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_tsfresh_dataset(
            tslearn.utils.to_tsfresh_dataset(tslearn_dataset)
        )
    )

    tslearn_dataset = rng.randn(1, sz, d)
    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_pyflux_dataset(
            tslearn.utils.to_pyflux_dataset(tslearn_dataset)
        )
    )


def test_timestamps_utils():
    """Tests for variable-timestep utility functions."""
    # to_timestamps: basic
    t = tslearn.utils.to_timestamps([0., 1., 2.])
    assert_allclose(t, [0., 1., 2.])
    assert t.dtype == float

    # to_timestamps: with trailing NaN preserved
    t = tslearn.utils.to_timestamps([0., 1., np.nan])
    assert_allclose(t[:2], [0., 1.])
    assert np.isnan(t[2])

    # to_timestamps: remove_nans
    t = tslearn.utils.to_timestamps([0., 1., np.nan], remove_nans=True)
    assert len(t) == 2

    # to_timestamps: non-monotonic raises
    with pytest.raises(ValueError):
        tslearn.utils.to_timestamps([0., 2., 1.])

    # to_timestamps: constant (equal consecutive values) raises
    with pytest.raises(ValueError):
        tslearn.utils.to_timestamps([0., 1., 1., 2.])

    # to_timestamps_dataset: variable-length
    ds = tslearn.utils.to_timestamps_dataset([[0., 1., 2.], [0., 2.]])
    assert ds.shape == (2, 3)
    assert_allclose(ds[0], [0., 1., 2.])
    assert_allclose(ds[1, :2], [0., 2.])
    assert np.isnan(ds[1, 2])

    # to_timestamps_dataset: uniform (2-D array passthrough)
    arr = np.array([[0., 1., 2.], [0., 1., 2.]])
    ds = tslearn.utils.to_timestamps_dataset(arr)
    assert_allclose(ds, arr)

    # to_timestamps_dataset: None passthrough
    assert tslearn.utils.to_timestamps_dataset(None) is None

    # check_timestamps_dataset: valid
    X = np.array([[[1.], [2.], [3.]], [[4.], [5.], [np.nan]]])
    t = np.array([[0., 1., 2.], [0., 1., np.nan]])
    result = tslearn.utils.check_timestamps_dataset(t, X)
    assert result.shape == (2, 3)

    # check_timestamps_dataset: shape mismatch raises
    with pytest.raises(ValueError):
        tslearn.utils.check_timestamps_dataset(np.array([[0., 1.]]), X)

    # check_timestamps_dataset: non-monotonic row raises
    t_bad = np.array([[0., 2., 1.], [0., 1., np.nan]])
    with pytest.raises(ValueError):
        tslearn.utils.check_timestamps_dataset(t_bad, X)


def test_conversions_cesium():
    pytest.importorskip('cesium')
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    tslearn_dataset = rng.randn(n, sz, d)

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_cesium_dataset(
            tslearn.utils.to_cesium_dataset(tslearn_dataset)
        )
    )


def test_check_array():
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=FutureWarning)
        tslearn.utils.check_array([[0]], force_all_finite=False)


def test_check_X_y():
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=FutureWarning)
        tslearn.utils.check_X_y([[0]], [0], force_all_finite=False)


def test_check_equal_size():
    assert tslearn.utils.check_equal_size([])
    assert tslearn.utils.check_equal_size([[0], [0]])
    assert not tslearn.utils.check_equal_size([[0], [0, 0]])
