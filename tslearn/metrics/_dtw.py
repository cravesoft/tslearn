"""DTW metric toolbox."""

from numba import njit

import numpy
from joblib import Parallel, delayed

from tslearn.backend import instantiate_backend
from tslearn.backend.pytorch_backend import HAS_TORCH
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.utils.utils import _ts_size

from ._masks import (
    GLOBAL_CONSTRAINT_CODE,
    _compute_mask,
    _njit_compute_mask
)
from .utils import (
    _njit_compute_path,
    _compute_path,
    _cdist_generic
)


def dtw(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return it.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.

    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> dtw([1, 2, 3], [1., 2., 2., 3.])
    0.0
    >>> dtw([1, 2, 3], [1., 2., 2., 3., 4.])
    1.0

    The PyTorch backend can be used to compute gradients:

    >>> import torch
    >>> s1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> s2 = torch.tensor([[3.0], [4.0], [-3.0]])
    >>> sim = dtw(s1, s2, be="pytorch")
    >>> print(sim)
    tensor(6.4807, grad_fn=<SqrtBackward0>)
    >>> sim.backward()
    >>> print(s1.grad)
    tensor([[-0.3086],
            [-0.1543],
            [ 0.7715]])

    >>> s1_2d = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    >>> s2_2d = torch.tensor([[3.0, 3.0], [4.0, 4.0], [-3.0, -3.0]])
    >>> sim = dtw(s1_2d, s2_2d, be="pytorch")
    >>> print(sim)
    tensor(9.1652, grad_fn=<SqrtBackward0>)
    >>> sim.backward()
    >>> print(s1_2d.grad)
    tensor([[-0.2182, -0.2182],
            [-0.1091, -0.1091],
            [ 0.5455,  0.5455]])

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """  # noqa: E501
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if s1.shape[0] == 0 or s2.shape[0] == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    global_constraint_ = GLOBAL_CONSTRAINT_CODE[global_constraint]

    if be.is_numpy:
        dtw_ = _njit_dtw
    else:
        dtw_ = _dtw
    return dtw_(s1, s2, global_constraint_, sakoe_chiba_radius, itakura_max_slope)


def dtw_path(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return both the path and the
    similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} (X_{i} - Y_{j})^2}

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> float(dist)
    0.0
    >>> float(dtw_path([1, 2, 3], [1., 2., 2., 3., 4.])[1])
    1.0

    See Also
    --------
    dtw : Get only the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets
    dtw_path_from_metric : Compute a DTW using a user-defined distance metric

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if s1.shape[0] == 0 or s2.shape[0] == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    global_constraint_ = GLOBAL_CONSTRAINT_CODE[global_constraint]

    if be.is_numpy:
        dtw_path_ = _njit_dtw_path
    else:
        dtw_path_ = _dtw_path
    dist, path = dtw_path_(
        s1,
        s2,
        global_constraint_,
        sakoe_chiba_radius,
        itakura_max_slope
    )
    return path, dist


def accumulated_matrix(s1, s2, mask, be=None):
    """Compute the DTW accumulated cost matrix score between two time series.

    It is not required that both time series share the same size, but they must
    be the same dimension.

    Parameters
    ----------
    s1 : array-like, shape=(sz1,) or (sz1, d)
        First time series.
    s2 : array-like, shape=(sz2,) or (sz2, d)
        Second time series.
    mask : array-like, shape=(sz1, sz2)
        Mask used to constrain the region of computation. Unconsidered cells must have False values.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix. Non computed cells due to masking have infinite value.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, be=be)
    s2 = to_time_series(s2, be=be)

    if be.is_numpy:
        compute_accumulated_matrix = _njit_accumulated_matrix
    else:
        compute_accumulated_matrix = _accumulated_matrix
    return compute_accumulated_matrix(s1, s2, mask)


def __make_accumulated_matrix(backend):

    def _accumulated_matrix_generic(s1, s2, mask):
        l1 = s1.shape[0]
        l2 = s2.shape[0]
        cum_sum = backend.full((l1 + 1, l2 + 1), backend.inf)
        cum_sum[0, 0] = 0.0

        for i in range(l1):
            for j in range(l2):
                if mask[i, j]:
                    dist = 0.0
                    for di in range(s1[i].shape[0]):
                        diff = s1[i][di] - s2[j][di]
                        dist += diff * diff
                    cum_sum[i + 1, j + 1] = dist
                    cum_sum[i + 1, j + 1] += min(
                        cum_sum[i, j + 1],
                        cum_sum[i + 1, j],
                        cum_sum[i, j]
                    )
        return cum_sum[1:, 1:]

    if backend is numpy:
        return njit(nogil=True)(_accumulated_matrix_generic)
    else:
        return _accumulated_matrix_generic


_njit_accumulated_matrix = __make_accumulated_matrix(numpy)
if HAS_TORCH:
    _accumulated_matrix = __make_accumulated_matrix(instantiate_backend("TorchBackend"))
else:
    _accumulated_matrix = _njit_accumulated_matrix


def __make_dtw(backend):
    if backend is numpy:
        compute_mask_ = _njit_compute_mask
        accumulated_matrix_ = _njit_accumulated_matrix
    else:
        compute_mask_ = _compute_mask
        accumulated_matrix_ = _accumulated_matrix

    def _dtw_generic(
        s1,
        s2,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
    ):
        mask = compute_mask_(s1.shape[0], s2.shape[0], global_constraint, sakoe_chiba_radius, itakura_max_slope)
        cum_sum = accumulated_matrix_(s1, s2, mask)
        return backend.sqrt(cum_sum[-1, -1])

    if backend is numpy:
        return njit(nogil=True)(_dtw_generic)
    else:
        return _dtw_generic

_njit_dtw = __make_dtw(numpy)
if HAS_TORCH:
    _dtw = __make_dtw(instantiate_backend("torch"))
else:
    _dtw = _njit_dtw


def __make_dtw_path(backend):
    if backend is numpy:
        compute_mask_ = _njit_compute_mask
        accumulated_matrix_ = _njit_accumulated_matrix
        compute_path_ = _njit_compute_path
    else:
        compute_mask_ = _compute_mask
        accumulated_matrix_ = _accumulated_matrix
        compute_path_ = _compute_path

    def _dtw_path_generic(
        s1,
        s2,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
    ):
        mask = compute_mask_(s1.shape[0], s2.shape[0], global_constraint, sakoe_chiba_radius, itakura_max_slope)
        cum_sum = accumulated_matrix_(s1, s2, mask)
        path = compute_path_(cum_sum)
        return backend.sqrt(cum_sum[-1, -1]), path

    if backend is numpy:
        return njit(nogil=True)(_dtw_path_generic)
    else:
        return _dtw_path_generic


_njit_dtw_path = __make_dtw_path(numpy)
if HAS_TORCH:
    _dtw_path = __make_dtw_path(instantiate_backend("torch"))
else:
    _dtw_path = _njit_dtw_path


def cdist_dtw(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    r"""Compute cross-similarity matrix using Dynamic Time Warping (DTW)
    similarity measure.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that time series share the same size, but they
    must be the same dimension.
    DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.

    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.

    dtype

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    cdist : array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.

    Examples
    --------
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]])
    array([[0., 1.],
           [1., 0.]])
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 3], [2, 3, 4, 5]])
    array([[0.        , 2.44948974],
           [1.        , 1.41421356]])

    See Also
    --------
    dtw : Get DTW similarity score

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """  # noqa: E501
    be = instantiate_backend(be, dataset1, dataset2)
    dataset1 = to_time_series_dataset(dataset1, be=be)
    if dataset2 is not None:
        dataset2 = to_time_series_dataset(dataset2, be=be)

    return _cdist_dtw(
        dataset1=dataset1,
        dataset2=dataset2,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
    )


def _cdist_dtw(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    if be is None:
        be = instantiate_backend(dataset1, dataset2)
    dtw_ = _njit_dtw if be.is_numpy else _dtw
    return _cdist_generic(
        dist_fun=dtw_,
        dataset1=dataset1,
        dataset2=dataset2,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
        compute_diagonal=False,
        global_constraint=GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
    )


# ---------------------------------------------------------------------------
# Time-aware DTW: handles irregular sampling via explicit timestamps
# ---------------------------------------------------------------------------

def __make_accumulated_matrix_with_timestamps(backend):
    """Factory producing an accumulated-cost-matrix function that penalises
    temporal misalignment in addition to value distance."""

    def _amwt(s1, s2, mask, t1, t2, time_weight):
        l1 = s1.shape[0]
        l2 = s2.shape[0]
        cum_sum = backend.full((l1 + 1, l2 + 1), backend.inf)
        cum_sum[0, 0] = 0.0
        for i in range(l1):
            for j in range(l2):
                if mask[i, j]:
                    dist = 0.0
                    for di in range(s1[i].shape[0]):
                        diff = s1[i][di] - s2[j][di]
                        dist += diff * diff
                    dt = t1[i] - t2[j]
                    dist += time_weight * dt * dt
                    cum_sum[i + 1, j + 1] = dist
                    cum_sum[i + 1, j + 1] += min(
                        cum_sum[i, j + 1],
                        cum_sum[i + 1, j],
                        cum_sum[i, j],
                    )
        return cum_sum[1:, 1:]

    if backend is numpy:
        return njit(nogil=True)(_amwt)
    else:
        return _amwt


_njit_accumulated_matrix_with_timestamps = (
    __make_accumulated_matrix_with_timestamps(numpy)
)
if HAS_TORCH:
    _accumulated_matrix_with_timestamps = (
        __make_accumulated_matrix_with_timestamps(instantiate_backend("torch"))
    )
else:
    _accumulated_matrix_with_timestamps = _njit_accumulated_matrix_with_timestamps


def _normalize_timestamps(t):
    """Normalize a 1-D timestamp array to [0, 1].

    If all values are identical the array is mapped to all-zeros.
    """
    t = numpy.asarray(t, dtype=float)
    t_min, t_max = t[0], t[-1]
    if t_max == t_min:
        return numpy.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


def dtw_with_timestamps(
    s1,
    s2,
    t1,
    t2,
    time_weight=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
):
    r"""Compute time-aware Dynamic Time Warping (DTW) between two time series.

    Extends DTW with a penalty term based on the actual timestamps of
    observations, making it aware of irregular sampling.  The local cost
    between observations *i* and *j* is:

    .. math::

        c(i, j) = \|X_i - Y_j\|^2
                  + \lambda \cdot (\hat{t}_{1,i} - \hat{t}_{2,j})^2

    where :math:`\lambda` is ``time_weight`` and :math:`\hat{t}` denotes
    timestamps normalised to [0, 1] within each series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        First time series.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Second time series.
    t1 : array-like, shape=(sz1,)
        Timestamps for s1. Must be strictly monotonically increasing.
    t2 : array-like, shape=(sz2,)
        Timestamps for s2. Must be strictly monotonically increasing.
    time_weight : float (default: 1.0)
        Weight :math:`\lambda` on the temporal penalty. Set to ``0.`` to
        recover standard DTW behaviour.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths.
    sakoe_chiba_radius : int or None (default: None)
        Radius for Sakoe-Chiba band constraint.
    itakura_max_slope : float or None (default: None)
        Maximum slope for Itakura parallelogram constraint.

    Returns
    -------
    float
        Time-aware DTW similarity score.

    Examples
    --------
    >>> dtw_with_timestamps([1, 2, 3], [1., 2., 3.], [0., 1., 2.], [0., 1., 2.])
    0.0
    >>> # time_weight=0 recovers standard DTW
    >>> dtw_with_timestamps([1, 2, 3], [1., 2., 2., 3.],
    ...                     [0., 1., 2.], [0., 1., 2., 3.], time_weight=0.)
    0.0

    See Also
    --------
    dtw : Standard DTW without timestamp awareness
    dtw_path_with_timestamps : Get both the path and the score
    cdist_dtw_with_timestamps : Cross-similarity matrix for datasets
    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if s1.shape[0] == 0 or s2.shape[0] == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )
    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    t1 = _normalize_timestamps(numpy.asarray(t1, dtype=float).ravel()[: s1.shape[0]])
    t2 = _normalize_timestamps(numpy.asarray(t2, dtype=float).ravel()[: s2.shape[0]])

    if len(t1) != s1.shape[0]:
        raise ValueError("t1 length must match s1 length.")
    if len(t2) != s2.shape[0]:
        raise ValueError("t2 length must match s2 length.")

    gc = GLOBAL_CONSTRAINT_CODE[global_constraint]
    mask = _njit_compute_mask(s1.shape[0], s2.shape[0], gc, sakoe_chiba_radius, itakura_max_slope)
    cum_sum = _njit_accumulated_matrix_with_timestamps(
        s1, s2, mask, t1, t2, float(time_weight)
    )
    return float(numpy.sqrt(cum_sum[-1, -1]))


def dtw_path_with_timestamps(
    s1,
    s2,
    t1,
    t2,
    time_weight=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
):
    r"""Compute time-aware DTW and return both the optimal path and score.

    See :func:`dtw_with_timestamps` for the full description of the
    time-aware cost formulation.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        First time series.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Second time series.
    t1 : array-like, shape=(sz1,)
        Timestamps for s1. Must be strictly monotonically increasing.
    t2 : array-like, shape=(sz2,)
        Timestamps for s2. Must be strictly monotonically increasing.
    time_weight : float (default: 1.0)
        Weight on the temporal penalty.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths.
    sakoe_chiba_radius : int or None (default: None)
        Radius for Sakoe-Chiba band constraint.
    itakura_max_slope : float or None (default: None)
        Maximum slope for Itakura parallelogram constraint.

    Returns
    -------
    list of integer pairs
        Optimal alignment path as (i, j) index pairs.
    float
        Time-aware DTW similarity score.

    Examples
    --------
    >>> path, dist = dtw_path_with_timestamps(
    ...     [1, 2, 3], [1., 2., 3.], [0., 1., 2.], [0., 1., 2.]
    ... )
    >>> path
    [(0, 0), (1, 1), (2, 2)]
    >>> float(dist)
    0.0

    See Also
    --------
    dtw_path : Standard DTW path without timestamp awareness
    dtw_with_timestamps : Get only the similarity score
    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if s1.shape[0] == 0 or s2.shape[0] == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )
    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    t1 = _normalize_timestamps(numpy.asarray(t1, dtype=float).ravel()[: s1.shape[0]])
    t2 = _normalize_timestamps(numpy.asarray(t2, dtype=float).ravel()[: s2.shape[0]])

    if len(t1) != s1.shape[0]:
        raise ValueError("t1 length must match s1 length.")
    if len(t2) != s2.shape[0]:
        raise ValueError("t2 length must match s2 length.")

    gc = GLOBAL_CONSTRAINT_CODE[global_constraint]
    mask = _njit_compute_mask(s1.shape[0], s2.shape[0], gc, sakoe_chiba_radius, itakura_max_slope)
    cum_sum = _njit_accumulated_matrix_with_timestamps(
        s1, s2, mask, t1, t2, float(time_weight)
    )
    path = _njit_compute_path(cum_sum)
    return path, float(numpy.sqrt(cum_sum[-1, -1]))


def cdist_dtw_with_timestamps(
    dataset1,
    timestamps1,
    dataset2=None,
    timestamps2=None,
    time_weight=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
):
    r"""Compute cross-similarity matrix using time-aware DTW.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz1, d)
        First dataset of time series.
    timestamps1 : array-like, shape=(n_ts1, sz1)
        Timestamps for dataset1. Each row must be strictly monotonically
        increasing (NaN-padded for variable-length series).
    dataset2 : array-like or None, shape=(n_ts2, sz2, d) (default: None)
        Second dataset. If None, self-similarity of dataset1 is returned.
    timestamps2 : array-like or None, shape=(n_ts2, sz2) (default: None)
        Timestamps for dataset2.
    time_weight : float (default: 1.0)
        Weight on the temporal penalty term.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths.
    sakoe_chiba_radius : int or None (default: None)
        Radius for Sakoe-Chiba band constraint.
    itakura_max_slope : float or None (default: None)
        Maximum slope for Itakura parallelogram constraint.
    n_jobs : int or None (default: None)
        Number of parallel jobs (passed to :class:`joblib.Parallel`).
    verbose : int (default: 0)
        Verbosity level.

    Returns
    -------
    numpy.ndarray, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.

    Examples
    --------
    >>> X = [[1, 2, 3], [1., 2., 4.]]
    >>> T = [[0., 1., 2.], [0., 1., 2.]]
    >>> cdist_dtw_with_timestamps(X, T)
    array([[0., 1.],
           [1., 0.]])

    See Also
    --------
    cdist_dtw : Standard DTW cross-similarity without timestamp awareness
    dtw_with_timestamps : Single-pair time-aware DTW
    """
    from tslearn.utils import check_timestamps_dataset, to_timestamps_dataset

    dataset1 = to_time_series_dataset(dataset1)
    ts1_arr = to_timestamps_dataset(timestamps1, max_sz=dataset1.shape[1])
    timestamps1_ = check_timestamps_dataset(ts1_arr, dataset1)

    self_similarity = dataset2 is None
    if self_similarity:
        dataset2 = dataset1
        timestamps2_ = timestamps1_
    else:
        dataset2 = to_time_series_dataset(dataset2)
        ts2_arr = to_timestamps_dataset(timestamps2, max_sz=dataset2.shape[1])
        timestamps2_ = check_timestamps_dataset(ts2_arr, dataset2)

    n_ts1 = dataset1.shape[0]
    n_ts2 = dataset2.shape[0]
    gc = GLOBAL_CONSTRAINT_CODE[global_constraint]

    def _pair(i, j):
        sz_i = _ts_size(dataset1[i])
        sz_j = _ts_size(dataset2[j])
        s1_ = dataset1[i, :sz_i]
        s2_ = dataset2[j, :sz_j]
        t1_ = _normalize_timestamps(timestamps1_[i, :sz_i])
        t2_ = _normalize_timestamps(timestamps2_[j, :sz_j])
        mask = _njit_compute_mask(sz_i, sz_j, gc, sakoe_chiba_radius, itakura_max_slope)
        cum_sum = _njit_accumulated_matrix_with_timestamps(
            s1_, s2_, mask, t1_, t2_, float(time_weight)
        )
        return float(numpy.sqrt(cum_sum[-1, -1]))

    if self_similarity:
        indices = [(i, j) for i in range(n_ts1) for j in range(i, n_ts1)]
    else:
        indices = [(i, j) for i in range(n_ts1) for j in range(n_ts2)]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_pair)(i, j) for i, j in indices
    )

    cdist = numpy.zeros((n_ts1, n_ts2))
    for (i, j), dist in zip(indices, results):
        cdist[i, j] = dist
        if self_similarity and i != j:
            cdist[j, i] = dist
    return cdist
