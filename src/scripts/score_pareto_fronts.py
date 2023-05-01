# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.10.8 ('main')
#     language: python
#     name: python3
# ---

import numpy as np
from matplotlib import pyplot as plt
from paths import srbench as srbench_path
from pathlib import Path
from os.path import basename, exists
from benchmark import gen_dataset, get_problem, load
from IPython.display import Latex, display
import textwrap

# In this file, we want to make it easy to go through a huge number of results, and have
# the user score whether the code found a particular equation.

dataset = "hubble"
data = load()

# +
all_algs = sorted(
    set(
        list(
            map(
                lambda p: basename(str(p)).split("_")[0],
                srbench_path.glob("*results/*results.txt"),
            )
        )
    )
)
all_datasets = (
    "hubble kepler newton tully leavitt schechter bode ideal_gas planck rydberg".split(
        " "
    )
)

all_seeds = list(range(1, 6))


# +
for dataset in all_datasets:
    for alg in all_algs:
        for seed in all_seeds:

            problem, name = get_problem(data, dataset)
            X, y = gen_dataset(data, dataset, scale_dataset=True)

            # display(Latex("$" + problem["equation"]["reduced_enum"] + "$"))
            # Take the highest version:
            def get_version_from_fname(x):
                # Converts ".../taken_results/E2ET_bode_seed=1_v48_duration.txt" to 48
                version = int(str(x).split("/")[-1].split("_")[-2][1:])
                if version == 999:
                    return 0
                return version

            fnames = sorted(
                map(
                    str,
                    srbench_path.glob(
                        f"*results/{alg}_{dataset}_seed={seed}*results*txt"
                    ),
                ),
                key=get_version_from_fname,
            )
            output_fname = (
                srbench_path / "analysis" / f"{alg}_{dataset}_seed={seed}.txt"
            )

            if exists(output_fname) and open(output_fname).read().strip() != "":
                print("Skipping", output_fname)
                continue

            with open(output_fname, "w") as f:
                if len(fnames) == 0:
                    print("incomplete", file=f)
                    continue

                fname = fnames[-1]

                version = get_version_from_fname(fname)
                if version < 44:
                    print("incomplete", file=f)
                    continue

                with open(fname, "r") as fr:
                    lines = fr.readlines()

                for i, line in enumerate(lines):
                    # print(line.replace("\n", ""))
                    # Print this line, indented (even if very long)
                    # Use: textwrap.wrap(line, 80)
                    for j, subline in enumerate(textwrap.wrap(line, 140)):
                        s = textwrap.indent(subline, " " * 4)
                        # Put the number on the first line:
                        if j == 0:
                            s = f"{i:3d}: " + s[4:]
                        print(s)

                version = get_version_from_fname(fname)
                print("We wanted to find")
                print("$" + problem["equation"]["reduced"] + "$", end=" or ")
                print("$" + problem["equation"]["reduced_enum"] + "$")
                print(
                    f"Was this search successful (for {alg}, seed={seed}, version={version})?"
                    "(yes, no, almost, incomp)"
                )
                # Get user input:
                user_input = input()
                assert user_input in ["yes", "no", "almost", "incomp"]
                print(user_input, file=f)
                print(basename(fname), file=f)
