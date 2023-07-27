# This example submission shows the submission of FEAT (cavalab.org/feat).
from pysr import PySRRegressor
import re
import os

try:
    num_cores = os.environ["OMP_NUM_THREADS"]
except KeyError:
    from multiprocessing import cpu_count

    num_cores = cpu_count()


warmup_time_in_minutes = 10
custom_operators = [
    "log",
    "sqrt",
]
standard_operators = [
    "square",
    "cube",
    "exp",
]

est = PySRRegressor(
    model_selection="best",
    loss="L1DistLoss()",
    procs=num_cores,
    progress=False,
    update=False,
    precision=64,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=standard_operators + custom_operators,
    maxsize=30,
    maxdepth=20,
    niterations=1000000,
    timeout_in_seconds=60 * (60 - warmup_time_in_minutes),
    # Warmup until 2% finished:
    warmup_maxsize_by=0.2 * 0.01,
    constraints={
        "square": 13,
        "cube": 13,
        "exp": 13,
        "log": 13,
        "sqrt": 13,
        "square": 13,
        "cube": 13,
        "/": (-1, 13),
        "*": (-1, -1),
        "+": (-1, -1),
        "-": (-1, -1),
    },
    nested_constraints={
        "/": {"/": 2},
        "exp": {"exp": 0, "log": 1, "sqrt": 0},
        "square": {"square": 1, "cube": 1, "log": 0, "sqrt": 0},
        "cube": {"cube": 1, "square": 1},
        "log": {"log": 0, "exp": 1},
        "sqrt": {"sqrt": 1, "exp": 1, "square": 0},
    },
)
# want to tune your estimator? wrap it in a sklearn CV class.


def find_parens(s):
    """Copied from https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python"""
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == "(":
            pstack.append(i)
        elif c == ")":
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret


def replace_prefix_operator_with_postfix(s, prefix, postfix_replacement):
    while re.search(prefix, s):
        parens_map = find_parens(s)
        # Find parentheses at start of prefix:
        start_model_str = re.search(prefix, s).span()[0]
        start_parens = re.search(prefix, s).span()[1]
        end_parens = parens_map[start_parens]
        s = (
            s[:start_model_str]
            + "("
            + s[start_parens : end_parens + 1]
            + postfix_replacement
            + ")"
            + s[end_parens + 1 :]
        )
    return s


def clean_model_str(model_str):
    # Replacements:
    # square(...) => (...)**2
    model_str = replace_prefix_operator_with_postfix(model_str, "square", "**2")
    # cube(x) => (x)**3
    model_str = replace_prefix_operator_with_postfix(model_str, "cube", "**3")

    return model_str


def model(est, X=None):
    """
    Return a sympy-compatible string of the final model.

    Parameters
    ----------
    est: sklearn regressor
        The fitted model.
    X: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A sympy-compatible string of the final model.
    """
    model_str = est.get_best().equation

    return model_str


def models(est, X=None):
    """
    Return the pareto front of sympy-compatible strings.

    Parameters
    ----------
    est: sklearn regressor
        The fitted model.
    X: pd.DataFrame, default=None
        The training data. This argument can be dropped if desired.

    Returns
    -------
    A list of sympy-compatible strings of the final model.
    """
    model_strs = est.equations_.equation
    model_strs = [clean_model_str(model_str) for model_str in model_strs]
    return model_strs


################################################################################
# Optional Settings
################################################################################


"""
eval_kwargs: a dictionary of variables passed to the evaluate_model()
    function. 
    Allows one to configure aspects of the training process.

Options 
-------
    test_params: dict, default = None
        Used primarily to shorten run-times during testing. 
        for running the tests. called as 
            est = est.set_params(**test_params)
    max_train_samples:int, default = 0
        if training size is larger than this, sample it. 
        if 0, use all training samples for fit. 
    scale_x: bool, default = True 
        Normalize the input data prior to fit. 
    scale_y: bool, default = True 
        Normalize the input label prior to fit. 
    pre_train: function, default = None
        Adjust settings based on training data. Called prior to est.fit. 
        The function signature should be (est, X, y). 
            est: sklearn regressor; the fitted model. 
            X: pd.DataFrame; the training data. 
            y: training labels.
"""


# define eval_kwargs.
eval_kwargs = dict(
    test_params=dict(
        populations=5,
        population_size=30,
        timeout_in_seconds=30,
    ),
)
