# Functions for loading the empirical equations dataset
import os

os.environ["MPLBACKEND"] = "agg"

import numpy as np
import mpmath as mp
import yaml
from typing import Dict
import pandas as pd
from hashlib import sha256
import ast

from paths import data as data_path

ebench_path = data_path / "empirical_bench"


def load():
    # Load YAML data:
    filename = ebench_path / "equations.yml"
    with open(filename, "r") as stream:
        data = yaml.safe_load(stream)
    return data


def exec_then_eval(code, rstate):
    block = ast.parse(code, mode="exec")

    # assumes last node is an expression
    last = ast.Expression(block.body.pop().value)

    _globals, _locals = {}, locals()
    exec(compile(block, "<string>", mode="exec"), _globals, _locals)
    return eval(compile(last, "<string>", mode="eval"), _globals, _locals)


def gen_dataset_using_generator(problem, rstate):
    data_generator = problem["data_generator"]

    args = data_generator["args"]
    str_args = ", ".join(args)

    # Define constants:
    if "constants" in data_generator:
        for constant, value in data_generator["constants"].items():
            if str(value) == "None":
                raise ValueError(f"Constant {constant} has value None")
            exec(str(constant) + " = mp.mpf('" + str(value) + "')", globals())

    callable = f"lambda {str_args}: {data_generator['callable']}"
    callable = eval(callable)

    size = int(data_generator["size"])
    X = {}
    if "ranges" in data_generator:
        for arg in args:
            low = float(data_generator["ranges"][arg][0])
            high = float(data_generator["ranges"][arg][1])

            if len(data_generator["ranges"][arg]) == 3:
                scale_type = str(data_generator["ranges"][arg][2])
            else:
                scale_type = "log"

            if scale_type == "log":
                X[arg] = rand_logarithmic(low, high, size, rstate=rstate)
            elif scale_type == "linear":
                X[arg] = rand_linear(low, high, size, rstate=rstate)
    else:
        custom_generation = data_generator["custom_generation"]
        # Init dictionary where every key is initialized as list:
        outputs = {}
        # cc = compile(custom_generation, "<string>", "exec")
        for i in range(size):
            output = exec_then_eval(custom_generation, rstate=rstate)
            # ^ Will redefine output.
            if i == 0:
                for key in output.keys():
                    outputs[key] = []
            for key in output.keys():
                outputs[key].append(output[key])
        for arg in outputs.keys():
            X[arg] = np.array(outputs[arg])

    # We do the computation using mpmath for precision.
    y = [callable(**{k: mp.mpf(v[i]) for k, v in X.items()}) for i in range(size)]
    y = np.array(y)

    assert "noise" in data_generator
    if "noise" in data_generator:
        if str(data_generator["noise"]["scale"]) == "None":
            # Default if nothing else is provided.
            scale = 0.1
        else:
            scale = float(data_generator["noise"]["scale"])

        if str(data_generator["noise"]["type"]) == "None":
            noise_type = "stdev"
        else:
            noise_type = data_generator["noise"]["type"]

        if "var" not in data_generator["noise"]:
            if noise_type == "absolute":
                y += rstate.randn(size) * scale
            elif noise_type == "relative":
                y += rstate.randn(size) * y * scale
            elif noise_type == "stdev":
                y += rstate.randn(size) * np.std(y) * scale
        else:
            col = data_generator["noise"]["var"]
            if noise_type == "absolute":
                X[col] += rstate.randn(size) * scale
            elif noise_type == "relative":
                X[col] += rstate.randn(size) * X[col] * scale
            elif noise_type == "stdev":
                X[col] += rstate.randn(size) * np.std(X[col]) * scale

    return pd.DataFrame(X), y


def gen_dataset_using_original(problem, dataset_key):
    # problem is a dict containing a "data"
    # key. This key is a dict containing "target"
    # and "inputs" keys. We create "y" based on "target"'s keys,
    # which are lists. We create a dataframe "X" based on "inputs"'s keys,
    # which are also lists.
    dataset = problem[dataset_key]
    if "csv" in dataset:
        #   csv:
        #     filename: leavitt.csv
        #     transform:
        #       P: 10 ** logP
        #       target: M
        filename = ebench_path / dataset["csv"]["filename"]
        transforms = dataset["csv"]["transform"]
        csv_data = pd.read_csv(filename)
        for key in csv_data.keys():
            # Bring variables into local space, e.g., logP = csv_data["logP"]
            exec(f"{key} = csv_data['{key}']")
        inputs = {}
        for key in transforms.keys():
            if key == "target":
                y = eval(transforms[key])
            else:
                inputs[key] = eval(transforms[key])
        X = pd.DataFrame(inputs)
    else:
        target = dataset["target"]
        inputs = dataset["inputs"]
        y = np.array([target[i] for i in range(len(target))])
        X = pd.DataFrame(inputs)
    return X, y


def get_problem(data, key):
    problems = data["problems"]
    problems: Dict
    # Match problem where problem['key'] == key:
    for name, problem in problems.items():
        if problem["key"] == key:
            break
    else:
        raise ValueError(f"Key {key} not found.")

    return problem, name


def gen_dataset(data, key, scale_dataset=False, return_info=False):
    problem, _ = get_problem(data, key)

    seed = np.frombuffer(sha256(key.encode()).digest(), dtype=np.uint32)
    rstate = np.random.RandomState(seed)

    if "use" in problem:
        dataset_key = problem["use"]
    else:
        dataset_key = "data" if "data" in problem else "data_generator"

    if dataset_key in ["data", "alternate_data"]:
        X, y = gen_dataset_using_original(problem, dataset_key)
    elif dataset_key == "data_generator":
        X, y = gen_dataset_using_generator(problem, rstate)

    log_scaling = False
    if (
        scale_dataset
        and "plot" in problem
        and "yscale" in problem["plot"]
        and problem["plot"]["yscale"] == "log"
    ):
        # See if problem y-axis is log-scaled, then do it on the data.
        # This is to allow the loss to fit all ranges of data.
        # We do NOT scale the input data - algorithms must deal with this.
        y = np.array([mp.log(yi) for yi in y])
        log_scaling = True

    if return_info:
        return X, y, {"log": log_scaling}
    return X, y


def rand_logarithmic(low, high, size, rstate):
    log_low = np.log(low)
    log_high = np.log(high)
    log_data = rstate.rand(size) * (log_high - log_low) + log_low
    data = np.exp(log_data)
    return data


def rand_linear(low, high, size, rstate):
    data = rstate.rand(size) * (high - low) + low
    return data
