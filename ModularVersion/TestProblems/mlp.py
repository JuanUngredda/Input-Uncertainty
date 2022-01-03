import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import get_scorer
from sklearn import datasets
from sklearn.utils import shuffle
import pickle 

from sbo.test_functions.space import JointSpace

CV_SPLITS = 5  # 5 probably makes more sense, but will be slower.

# MLP with ADAM
mlp_adam_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "beta_1": {"type": "real", "space": "logit", "range": (0.5, 0.99)},
    "beta_2": {"type": "real", "space": "logit", "range": (0.9, 1.0 - 1e-6)},
    "epsilon": {"type": "real", "space": "log", "range": (1e-9, 1e-6)},
}

# random_state = None => just use global random stream for now
# warm_start = False, to ensure each run to fit is an indep func call, otherwise violates BO assumptions
mlp_adam_fixed = {"random_state": None, "verbose": False, "warm_start": False, "solver": "adam", "early_stopping": True}


mlp_gen_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},  # TODO generalize with depth
    "activation": {"type": "cat", "values": ("identity", "logistic", "tanh", "relu")},
    "solver": {"type": "cat", "values": ("lbfgs", "sgd", "adam")},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate": {"type": "cat", "values": ("constant", "invscaling", "adaptive")},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "power_t": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "max_iter": {"type": "int", "space": "log", "range": (50, 1000)},
    "shuffle": {"type": "bool"},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "momentum": {"type": "real", "space": "logit", "range": (0.001, 0.999)},
    "nesterovs_momentum": {"type": "bool"},
    "early_stopping": {"type": "bool"},
    "validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "beta_1": {"type": "real", "space": "logit", "range": (0.5, 0.99)},
    "beta_2": {"type": "real", "space": "logit", "range": (0.9, 1.0 - 1e-6)},
    "epsilon": {"type": "real", "space": "log", "range": (1e-9, 1e-6)},
    "n_iter_no_change": {"type": "int", "space": "log", "range": (1, 100)},
}

# random_state = None => just use global random stream for now
# warm_start = False, to ensure each run to fit is an indep func call, otherwise violates BO assumptions
mlp_gen_fixed = {"random_state": 123, "verbose": False, "warm_start": False}


class NeuralNetwork:
    def __init__(self, shuffle_seed=0):
        self.base_model = MLPClassifier
        self.fixed_params = mlp_gen_fixed

        self.space_x = JointSpace(mlp_gen_cfg)
        bounds = self.space_x.get_bounds()

        self.lb = bounds[:, 0]  # In warped space
        self.ub = bounds[:, 1]  # In warped space
        self.dim = bounds.shape[0]

        metric = "accuracy"
        self.scorer = get_scorer(metric)

        # Now setup data set
        data, target = datasets.load_digits(return_X_y=True)
        # data = pickle.load(open("cifar10_X.pkl", "rb"))
        # target = pickle.load(open("cifar10_y.pkl", "rb"))
        # data, target = data[:10000, :], target[:10000]

        # Do some validation on loaded data
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert data.ndim == 2 and target.ndim == 1
        assert data.shape[0] == target.shape[0]
        assert data.size > 0
        assert data.dtype == np.float_
        assert np.all(np.isfinite(data))  # also catch nan
        assert target.dtype == np.int_
        assert np.all(np.isfinite(target))  # also catch nan

        # Always shuffle your data to be safe. Use fixed seed for reprod.
        self.data_X, self.data_y = shuffle(data, target, random_state=shuffle_seed)

    def __call__(self, x):
        try:
            assert x.shape == (self.dim,)
            x = np.clip(x, a_min=self.lb, a_max=self.ub)

            params, = self.space_x.unwarp([x])
            params.update(self.fixed_params)  # add in fixed params

            # now build the skl object
            clf = self.base_model(**params)

            assert np.all(np.isfinite(self.data_X)), "all features must be finite"
            assert np.all(np.isfinite(self.data_y)), "all targets must be finite"

            # Do the x-val, ignore user warn since we expect BO to try weird stuff
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                S = cross_val_score(clf, self.data_X, self.data_y, scoring=self.scorer, cv=CV_SPLITS)
            # Take the mean score across all x-val splits
            overall_score = np.mean(S)

            # get_scorer makes everything a score not a loss, so we need to negate to get the loss back.
            assert np.isfinite(overall_score), "loss not even finite"

            # Evaluate and return value
            return -overall_score  # Since we are minimizing
        except:
            print("Warning: Crashed")
            return 3.0  # Seems reasonable for MNIST + logloss
