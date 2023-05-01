"""Evaluate a given method on a given dataset.

(Assumes that the corresponding conda environment is activated.)
"""
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
from local_paths import data_path, methods_path, src_path, results_path

os.environ["OMP_NUM_THREADS"] = "8"

sys.path.append(str(src_path / "scripts"))

from benchmark import gen_dataset, get_problem, load

if __name__ == "__main__":
    argparser = ArgumentParser(description=__doc__)
    argparser.add_argument("--method", help="method to evaluate", type=str)
    argparser.add_argument("--dataset", help="dataset to evaluate", type=str)
    argparser.add_argument("--seed", help="seed", type=int)
    argparser.add_argument("--version", help="version", type=int)
    argparser.add_argument("--test", help="just testing", action="store_true")

    args = argparser.parse_args()
    np.random.seed(args.seed)
    method_path = methods_path / args.method

    sys.path.append(str(method_path))

    data = load()
    problem, name = get_problem(data, args.dataset)
    X, y = gen_dataset(data, args.dataset, scale_dataset=True)

    from regressor import est, eval_kwargs

    try:
        from regressor import models
    except:
        from regressor import model

        models = lambda *args, **kwargs: [model(*args, **kwargs)]

    X_values = np.array(X.values, dtype=np.float64)
    y_values = np.array([float(elem) for elem in y], dtype=np.float64)

    if "test_params" in eval_kwargs and args.test:
        est.set_params(**eval_kwargs["test_params"])
    elif "pre_train" in eval_kwargs and not args.test:
        eval_kwargs["pre_train"](est, X_values, y_values)

    filename_base = str(
        results_path / f"{args.method}_{args.dataset}_seed={args.seed}_v{args.version}"
    )

    start = time()
    est.fit(X_values, y_values)
    duration = time() - start

    model_strs = models(est, X)

    with open(filename_base + "_duration.txt", "w") as f:
        f.write(f"{duration:.2f}")

    # Write these to a file:
    with open(filename_base + "_results.txt", "w") as f:
        f.write("\n".join(model_strs))
