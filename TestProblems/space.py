import numpy as np
from scipy.interpolate import interp1d
from scipy.special import expit as logistic  # because nobody calls it expit
from scipy.special import logit

WARPED_DTYPE = np.float_
N_GRID_DEFAULT = 8

# I can't make up mind of unicode or str is better wrt to Py 2/3 compatibility
# ==> Just make a global constant and make sure it works either way.
# Note: if we switch to np.str_, we will also need to update doc-strings!
CAT_DTYPE = np.unicode_
CAT_KIND = "U"
CAT_NATIVE_DTYPE = str
# Check to make sure consistent
assert CAT_KIND == np.dtype(CAT_DTYPE).kind
_infered = type(CAT_DTYPE("").item())
assert CAT_NATIVE_DTYPE == _infered

# Skip these for now to minimize deps:


def check_array(*args, **kwargs):
    pass


clip_chk = np.clip

# ============================================================================
# These could go into util
# ============================================================================


def unravel_index(dims):
    """Builds tuple of coordinate arrays to traverse an numpy array.

    Wrapper around `np.unravel_index` that avoids bug at corner case for ``dims=()``. The fix for this has been merged
    into the numpy master branch Oct 18, 2017 so future numpy releases will make this wrapper not needed. Otherwise,
    ``unravel_index(X.shape)`` is equivalent to:``np.unravel_index(range(X.size), X.shape)``.

    Parameters
    ----------
    dims : tuple of ints
        The shape of the array to use for unraveling ``indices``.

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has shape (n,) where ``n=np.prod(dims)``.

    References
    ----------
    unravel_index(0, ()) should return () (Trac #2120) #580
    https://github.com/numpy/numpy/issues/580
    Allow `unravel_index(0, ())` to return () #9884
    https://github.com/numpy/numpy/pull/9884
    """
    size = np.prod(dims)
    if dims == () or size == 0:  # The corner case
        return ()

    idx = np.unravel_index(range(np.prod(dims)), dims)
    return idx


def encode(X, labels, assume_sorted=False, dtype=bool, assume_valid=False):
    """Perform one hot encoding of categorical data in array-like variable `X` of any dimension.

    Parameters
    ----------
    X : array-like, shape (...)
        Categorical values of any standard type. Vectorized to work for any dimensional `X`.
    labels : array-like, shape (n,)
        Complete list of all possible labels. List is flattened if it is not already 1 dimensional.
    assume_sorted : bool
        If True, assume labels is already sorted and unique. This saves the computational cost of calling `np.unique`.
    dtype : dtype
        Desired data of feature array. One-hot is most logically `bool`, but feature matrices are usually `float`.
    assume_valid : bool
        If True, assume all element of `X` are in the list `labels`. This saves the computational cost of verifying
        `X` are in `labels`. If True and a non-label `X` occurs this routine will silently give bogus result.

    Returns
    -------
    Y : ndarray, shape (..., n)
        One-hot encoding of `X`. Extra dimension is appended at end for the one-hot vector. It has data type `dtype`.
    """
    X = np.asarray(X)
    labels = np.asarray(labels) if assume_sorted else np.unique(labels)
    check_array(labels, "labels", pre=True, ndim=1, min_size=1)

    idx = np.searchsorted(labels, X)
    # If x is not even in labels then this will fail. This is not ValueError
    # because the user explictly asked for this using argument assume_valid.
    assert assume_valid or np.all(np.asarray(labels[idx]) == X)

    # This is using some pro np indexing technique to vectorize across all
    # possible input dimensions for X in the same code.
    Y = np.zeros(X.shape + (len(labels),), dtype=dtype)
    Y[unravel_index(X.shape) + (idx.ravel(),)] = True
    return Y


def decode(Y, labels, assume_sorted=False):
    """Perform inverse of one-hot encoder `encode`.

    Parameters
    ----------
    Y : ndarray, shape (..., n)
        One-hot encoding of categorical data `X`. Extra dimension is appended at end for the one-hot vector. Maximum
        element is taken if there is more than one non-zero entry in one-hot vector.
    labels : array-like, shape (n,)
        Complete list of all possible labels. List is flattened if it is not already 1 dimensional.
    assume_sorted : bool
        If True, assume labels is already sorted and unique. This saves the computational cost of calling `np.unique`.

    Returns
    -------
    X : array-like, shape (...)
        Categorical values corresponding to one-hot encoded `Y`.
    """
    Y = np.asarray(Y)
    labels = np.asarray(labels) if assume_sorted else np.unique(labels)
    check_array(labels, "labels", pre=True, ndim=1, min_size=1)
    check_array(Y, "Y", pre=True, shape_endswith=(len(labels),))

    idx = np.argmax(Y, axis=-1)
    X = labels[idx]
    return X


# ============================================================================
# Setup warping dictionaries
# ============================================================================


def identity(x):
    """Helper function that perform warping in linear space. Sort of a no-op.

    Parameters
    ----------
    x : scalar
        Input variable in linear space. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : float
        Same as input `x`.
    """
    y = x
    return y


def bilog(x):
    """Bilog warping function. Extension of log to work with negative numbers.

    Bilog(x) ~= log(x) for large x or -log(abs(x)) if x is negative. However, the bias term ensures good behavior
    near 0 and bilog(0) = 0.

    Parameters
    ----------
    x : scalar
        Input variable in linear space. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : float
        The bilog of `x`.
    """
    y = np.sign(x) * np.log(1.0 + np.abs(x))
    return y


def biexp(x):
    """Inverse of bilog function.

    Parameters
    ----------
    x : scalar
        Input variable in linear space. Can be any numeric type and is vectorizable.

    Returns
    -------
    y : float
        The biexp of `x`.
    """
    y = np.sign(x) * (np.exp(np.abs(x)) - 1.0)
    return y


WARP_DICT = {"linear": identity, "log": np.log, "logit": logit, "bilog": bilog}
UNWARP_DICT = {"linear": identity, "log": np.exp, "logit": logistic, "bilog": biexp}

# ============================================================================
# Setup spaces class hierarchy
# ============================================================================


class Space(object):
    def __init__(self, dtype, default_round, warp="linear", values=None, range_=None):
        """Generic constructor of `Space` class.

        Not intended to be called directly but instead by child classes. However, `Space` is not an abstract class and
        will not give an error when instantiated.
        """
        self.dtype = dtype
        assert warp in WARP_DICT, "invalid space %s, allowed spaces are: %s" % (str(warp), str(WARP_DICT.keys()))
        self.warp_f = WARP_DICT[warp]
        self.unwarp_f = UNWARP_DICT[warp]

        # Setup range and rounding if values is suplied
        assert (values is None) != (range_ is None)
        round_to_values = default_round
        if range_ is None:  # => value is not None
            # Debatable if unique should be done before or after cast. But I
            # think after is better, esp. when changing precisions.
            values = np.asarray(values, dtype=dtype)
            values = np.unique(values)  # values now 1D ndarray no matter what
            check_array(
                values,
                "unique values",
                pre=True,
                ndim=1,
                dtype=dtype,
                min_size=2,
                allow_infinity=False,
                allow_nan=False,
            )

            # Extrapolation might happen due to numerics in type conversions.
            # Bounds checking is still done in validate routines.
            round_to_values = interp1d(values, values, kind="nearest", fill_value="extrapolate")
            range_ = (values[0], values[-1])
        # Save values and rounding
        # Values is either None or was validated inside if statement
        self.values = values
        self.round_to_values = round_to_values

        # Note that if dtype=None that is the default for asarray.
        range_ = np.asarray(range_, dtype=dtype)
        check_array(range_, "range", pre=True, shape=(2,), dtype=dtype, unsorted=False)
        # Save range info, with input validation and post validation
        self.lower, self.upper = range_

        # Convert to warped bounds too with lots of post validation
        self.lower_warped, self.upper_warped = self.warp_f(range_[..., None]).astype(WARPED_DTYPE, copy=False)
        check_array(
            self.lower_warped,
            "warped lower bound %s(%.1f)" % (warp, self.lower),
            ndim=1,
            pre=True,
            dtype=WARPED_DTYPE,
            allow_infinity=False,
            allow_nan=False,
        )
        # Should never happen if warpers are strictly monotonic:
        assert np.all(self.lower_warped <= self.upper_warped)

        # Make sure a bit bigger to keep away from lower due to numerics
        self.upper_warped = np.maximum(self.upper_warped, np.nextafter(self.lower_warped, np.inf))
        check_array(
            self.upper_warped,
            "warped upper bound %s(%.1f)" % (warp, self.upper),
            pre=True,
            shape=self.lower_warped.shape,
            dtype=WARPED_DTYPE,
            allow_infinity=False,
            allow_nan=False,
        )
        # Should never happen if warpers are strictly monotonic:
        assert np.all(self.lower_warped < self.upper_warped)

    def validate(self, X, pre=False):
        """Routine to validate inputs to warp.

        This routine does not perform any checking on the dimensionality of `X` and is fully vectorized.
        """
        X = np.asarray(X, dtype=self.dtype)

        if self.values is None:
            X = clip_chk(X, self.lower, self.upper)
        else:
            check_array(X, "X", pre=pre, whitelist=self.values)

        return X

    def validate_warped(self, X, pre=False):
        """Routine to validate inputs to unwarp. This routine is vectorized, but `X` must have at least 1-dimension.
        """
        X = np.asarray(X, dtype=WARPED_DTYPE)
        check_array(X, "X", pre=pre, shape_endswith=(len(self.lower_warped),))

        X = clip_chk(X, self.lower_warped, self.upper_warped)
        return X

    def warp(self, X):
        """Warp inputs to a continous space.

        Parameters
        ----------
        X : array-like, shape (...)
            Input variables to warp. This is vectorized to work in any dimension, but it must have the same type code
            as the class, which is in `self.type_code`.

        Returns
        -------
        X_w : ndarray, shape (..., m)
            Warped version of input space. By convention there is an extra dimension on warped array.
            Currently, ``m=1`` for all warpers. `X_w` will have a float type.
        """
        X = self.validate(X, pre=True)

        X_w = self.warp_f(X)
        X_w = X_w[..., None]  # Convention is that warped has extra dim

        X_w = self.validate_warped(X_w)  # Ensures of WAPRED_DTYPE
        check_array(X_w, "X", ndim=X.ndim + 1, dtype=WARPED_DTYPE)
        return X_w

    def unwarp(self, X_w):
        """Inverse of `warp` function.

        Parameters
        ----------
        X_w : ndarray, shape (..., m)
            Warped version of input space. This is vectorized to work in any dimension. But, by convention, there is an
            extra dimension on the warped array. Currently, the last dimension ``m=1`` for all warpers. `X_w` must be of
            a float type.

        Returns
        -------
        X : array-like, shape (...)
            Unwarped version of `X_w`. `X` will have the same type code as the class, which is in `self.type_code`.
        """
        X_w = self.validate_warped(X_w, pre=True)

        X = clip_chk(self.unwarp_f(X_w[..., 0]), self.lower, self.upper)
        X = self.round_to_values(X)

        X = self.validate(X)  # Ensures of dtype
        check_array(X, "X", ndim=X_w.ndim - 1, dtype=self.dtype)
        return X

    def get_bounds(self):
        """Get bounds of the warped space.

        Returns
        -------
        bounds : ndarray, shape (D, 2)
            Bounds in the warped space. First column is the lower bound and the second column is the upper bound.
            ``bounds.tolist()`` gives the bounds in the standard form expected by scipy optimizers:
            ``[(lower_1, upper_1), ..., (lower_n, upper_n)]``.
        """
        bounds = np.stack((self.lower_warped, self.upper_warped), axis=1)
        check_array(bounds, "bounds", shape=(len(self.lower_warped), 2), dtype=WARPED_DTYPE)
        return bounds

    def grid(self, max_interp=N_GRID_DEFAULT):
        """Return grid spanning the original (unwarped) space.

        Parameters
        ----------
        max_interp : int
            The number of points to use in grid space when a range and not values are used to define the space.
            Must be >= 0.

        Returns
        -------
        values : list
            Grid spanning the original space. This is simply `self.values` if a grid has already been specified,
            otherwise it is just grid across the range.
        """
        values = self.values
        if values is None:
            vw = np.linspace(self.lower_warped, self.upper_warped, max_interp)
            # Some spaces like int make result in duplicates after unwarping
            # so we apply unique to avoid this. However this will usually be
            # wasted computation.
            values = np.unique(self.unwarp(vw[:, None]))
            check_array(values, "values", ndim=1, dtype=self.dtype)

        # Best to convert to list to make sure in native type
        values = values.tolist()
        return values


class Real(Space):
    def __init__(self, warp="linear", values=None, range_=None):
        """Build Real space class.

        Parameters
        ----------
        warp : {'linear', 'log', 'logit', 'bilog'}
            Which warping type to apply to the space. The warping is applied in the original space. That is, in a space
            with ``warp='log'`` and ``range_=(2.0, 10.0)``, the value 2.0 warps to log(2), not -inf as in some
            other frameworks.
        values : None or list-like of float
            Possible values for space to take. Values must be of float type.
        range_ : None or array-like of shape (2,)
            Array with (lower, upper) pair with limits of space. Note that one must specify `values` or `range_`, but
            not both. `range_` must be composed of floats.
        """
        # TODO this pre-check can be removed once we have API validator
        assert warp is not None, "warp/space not specified for real"
        Space.__init__(self, np.float_, identity, warp, values, range_)


class Integer(Space):
    def __init__(self, warp="linear", values=None, range_=None):
        """Build Integer space class.

        Parameters
        ----------
        warp : {'linear', 'log', 'bilog'}
            Which warping type to apply to the space. The warping is applied in the original space. That is, in a space
            with ``warp='log'`` and ``range_=(2, 10)``, the value 2 warps to log(2), not -inf as in some other
            frameworks. There are no settings with integers that are compatible with the logit warp.
        values : None or list-like of float
            Possible values for space to take. Values must be of int type.
        range_ : None or array-like of shape (2,)
            Array with (lower, upper) pair with limits of space. Note that one must specify `values` or `range_`, but
            not both. `range_` must be composed of ints.
        """
        # TODO this pre-check can be removed once we have API validator
        assert warp is not None, "warp/space not specified for int"
        Space.__init__(self, np.int_, np.round, warp, values, range_)


class Boolean(Space):
    def __init__(self, warp=None, values=None, range_=None):
        """Build Boolean space class.

        Parameters
        ----------
        warp : None
            Must be omitted or None, provided for consitency with other types.
        values : None
            Must be omitted or None, provided for consitency with other types.
        range_ : None
            Must be omitted or None, provided for consitency with other types.
        """
        assert warp is None, "cannot warp bool"
        assert (values is None) and (range_ is None), "cannot pass in values or range for bool"
        self.dtype = np.bool_
        self.warp_f = identity
        self.unwarp_f = identity

        self.values = np.array([False, True], dtype=np.bool_)
        self.round_to_values = np.round

        self.lower, self.upper = self.dtype(False), self.dtype(True)
        self.lower_warped = np.array([0.0], dtype=WARPED_DTYPE)
        self.upper_warped = np.array([1.0], dtype=WARPED_DTYPE)


class Categorical(Space):
    def __init__(self, warp=None, values=None, range_=None):
        """Build Integer space class.

        Parameters
        ----------
        warp : None
            Must be omitted or None, provided for consitency with other types.
        values : list-like of unicode
            Possible values for space to take. Values must be unicode strings. Requiring type unicode ('U') rather
            than strings ('S') provides better forward compatibility with Python 3.
        range_ : None
            Must be omitted or None, provided for consitency with other types.
        """
        assert warp is None, "cannot warp cat"
        assert values is not None, "must pass in explicit values for cat"
        assert range_ is None, "cannot pass in range for cat"

        values = np.unique(values)  # values now 1D ndarray no matter what
        check_array(values, "values", pre=True, ndim=1, kind=CAT_KIND, min_size=2)
        self.values = values

        self.dtype = CAT_DTYPE
        # Debatable if decode should go in unwarp or round_to_values
        self.warp_f = lambda x: encode(x, self.values, True, WARPED_DTYPE, True)
        self.unwarp_f = identity
        self.round_to_values = lambda y: decode(y, self.values, True)

        self.lower, self.upper = None, None  # Don't need them
        self.lower_warped = np.zeros(len(values), dtype=WARPED_DTYPE)
        self.upper_warped = np.ones(len(values), dtype=WARPED_DTYPE)

    def warp(self, X):
        """Warp inputs to a continous space.

        Parameters
        ----------
        X : array-like, shape (...)
            Input variables to warp. This is vectorized to work in any dimension, but it must have the same
            type code as the class, which is unicode ('U') for the `Categorical` space.

        Returns
        -------
        X_w : ndarray, shape (..., m)
            Warped version of input space. By convention there is an extra dimension on warped array. The warped space
            has a one-hot encoding and therefore `m` is the number of possible values in the space. `X_w` will have
            a float type.
        """
        X = self.validate(X, pre=True)

        X_w = self.warp_f(X)

        # Probably over kill to validate here too, but why not:
        X_w = self.validate_warped(X_w)
        check_array(X_w, "X", ndim=X.ndim + 1, dtype=WARPED_DTYPE)
        return X_w

    def unwarp(self, X_w):
        """Inverse of `warp` function.

        Parameters
        ----------
        X_w : ndarray, shape (..., m)
            Warped version of input space. The warped space has a one-hot  encoding and therefore `m` is the number of
            possible values in the space. `X_w` will have a float type. Non-zero/one values are allowed in `X_w`.
            The maximal element in the vector is taken as the encoded value.

        Returns
        -------
        X : array-like, shape (...)
            Unwarped version of `X_w`. `X` will have same type code as the `Categorical` class, which is unicode ('U').
        """
        X_w = self.validate_warped(X_w, pre=True)

        X = self.round_to_values(self.unwarp_f(X_w))

        X = self.validate(X)
        check_array(X, "X", ndim=X_w.ndim - 1, kind=CAT_KIND)
        return X


# Treat ordinal identically to categorical for now
SPACE_DICT = {"real": Real, "int": Integer, "bool": Boolean, "cat": Categorical, "ordinal": Categorical}

# ============================================================================
# Setup code for joint spaces over multiple parameters with different configs
# ============================================================================


class JointSpace(object):
    def __init__(self, meta):
        """Build Real space class.

        Parameters
        ----------
        meta : dict-like of dict-like
            Configuration of variables in joint space. See API description.
        """
        assert len(meta) > 0  # Unclear what to do with empty space

        # Lock in an order if not ordered dict, sorted helps reproducibility
        self.param_list = sorted(meta.keys())

        # TODO longer term let's have a seperate validation routine that
        # validates the API with pretty error messages, that way we can ignore
        # putting too much thought into the assertions msgs deep in the code.
        for param, config in meta.items():
            assert config["type"] in SPACE_DICT, "invalid input type %s, allowed types: %s" % (
                config["type"],
                str(SPACE_DICT.keys()),
            )

        spaces = {
            param: SPACE_DICT[config["type"]](
                config.get("space", None), config.get("values", None), config.get("range", None)
            )
            for param, config in meta.items()
        }
        self.spaces = spaces

        self.blocks = np.cumsum([len(spaces[param].get_bounds()) for param in self.param_list])

    def validate(self, X):
        """Raise `ValueError` if X does not match the format expected for a
        joint space."""
        for record in X:
            if self.param_list != sorted(record.keys()):
                raise ValueError("Expected joint space keys %s, but got %s", (self.param_list, sorted(record.keys())))
            for param in self.param_list:
                self.spaces[param].validate([record[param]], pre=True)
        # Return X back so we have option to cast it to list or whatever later
        return X

    def warp(self, X):
        """Warp inputs to a continous space.

        Parameters
        ----------
        X : list of dict-like, of shape (n,)
            List of `n` points in the joint space to warp. Each list element is a dictionary where each key corresponds
            to a variable in the joint space.

        Returns
        -------
        X_w : ndarray, shape (n, m)
            Warped version of input space. Result is 2D float np array. `n` is the number of input points, length
            of `X`. `m` is the size of the joint warped space, which can be inferred by calling `get_bounds`.
        """
        # It would be nice to have cleaner way to deal with this corner case
        if len(X) == 0:
            return np.zeros((0, self.blocks[-1]), dtype=WARPED_DTYPE)

        X_w = [np.concatenate([self.spaces[param].warp(record[param]) for param in self.param_list]) for record in X]
        X_w = np.stack(X_w, axis=0)
        check_array(X_w, "X", shape=(len(X), self.blocks[-1]), dtype=WARPED_DTYPE)
        return X_w

    def unwarp(self, X_w):
        """Inverse of `warp`.

        Parameters
        ----------
        X_w : array-like, shape (n, m)
            Warped version of input space. Must be 2D float array-like. `n` is the number of seperate points in the
            warped joint space. `m` is the size of the joint warped space, which can be inferred in advance by
            calling `get_bounds`.

        Returns
        -------
        X : list of dict-like, of shape (n,)
            List of `n` points in the joint space to warp. Each list element is a dictionary where each key corresponds
            to a variable in the joint space.
        """
        X_w = np.asarray(X_w)
        check_array(X_w, "X", ndim=2, shape_endswith=(self.blocks[-1],), dtype=WARPED_DTYPE)
        N = X_w.shape[0]

        X = {
            param: self.spaces[param].unwarp(xx) for param, xx in zip(self.param_list, np.hsplit(X_w, self.blocks[:-1]))
        }
        # Convert dict of arrays to list of dicts, this would not be needed if
        # we used pandas but we do not want to add it as a dep. np.asscalar and
        # .item() appear to be the same thing but asscalar seems more readable.
        X = [{param: X[param][ii].item() for param in self.param_list} for ii in range(N)]
        return X

    def get_bounds(self):
        """Get bounds of the warped joint space.

        Returns
        -------
        bounds : ndarray, shape (m, 2)
            Bounds in the warped space. First column is the lower bound and the second column is the upper bound.
            ``bounds.tolist()`` gives the bounds in the standard form expected by scipy optimizers:
            ``[(lower_1, upper_1), ..., (lower_n, upper_n)]``.
        """
        bounds = np.concatenate([self.spaces[param].get_bounds() for param in self.param_list], axis=0)
        check_array(bounds, "bounds", shape_endswith=(2,), dtype=WARPED_DTYPE)
        return bounds

    def grid(self, max_interp=N_GRID_DEFAULT):
        """Return grid spanning the original (unwarped) space.

        Parameters
        ----------
        max_interp : int
            The number of points to use in grid space when a range and not values are used to define the space.
            Must be >= 0.

        Returns
        -------
        axes : dict of list
            Grids spanning the original spaces of each variable. For each variable, this is simply `self.values`
            if a grid has already been specified, otherwise it is just grid across the range.
        """
        axes = {var_name: space.grid(max_interp=max_interp) for var_name, space in self.spaces.items()}
        return axes


if __name__ == "__main__":
    M = {
        "kernel": {"type": "cat", "values": ("rbf", "poly", "linear")},
        "C": {"type": "real", "space": "log", "range": (1.0, 1000.0)},
        "gamma": {"type": "real", "space": "log", "range": (1e-4, 1e-3)},
        "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
        "decision_function_shape": {"type": "cat", "values": ("ovo", "ovr")},
        "max_iter": {"type": "int", "space": "log", "range": (50, 1000)},
    }

    S = JointSpace(M)

    L = [
        {"kernel": "rbf", "C": 5.0, "gamma": 5e-4, "tol": 1e-4, "decision_function_shape": "ovo", "max_iter": 60},
        {"kernel": "linear", "C": 5.0, "gamma": 3e-4, "tol": 1e-4, "decision_function_shape": "ovr", "max_iter": 600},
    ]

    print("initial params")
    print(L)

    print("warped params")
    W = S.warp(L)
    print(W)

    print("unwarped params")
    L = S.unwarp(W)
    print(L)

    print("bounds on warped space")
    B = S.get_bounds()
    print(B)
