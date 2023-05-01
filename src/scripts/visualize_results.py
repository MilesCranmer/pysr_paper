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
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from paths import srbench as srbench_path, output as output_path
from os.path import basename, exists
from equations import load

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
short_name = {
    "hubble": "Hubble",
    "kepler": "Kepler",
    "newton": "Newton",
    "tully": "Tully",
    "leavitt": "Leavitt",
    "schechter": "Schechter",
    "bode": "Bode",
    "ideal_gas": "Ideal Gas",
    "planck": "Planck",
    "rydberg": "Rydberg",
}

all_seeds = list(range(1, 6))

midx = pd.MultiIndex.from_product([all_algs, all_datasets, all_seeds])
data = pd.DataFrame(index=midx, columns=["result", "file"])

# e.g.,
# data.loc["Bingo", "hubble", 1]
# -

for alg in all_algs:
    for dataset in all_datasets:
        for seed in all_seeds:
            fname = srbench_path / "analysis" / f"{alg}_{dataset}_seed={seed}.txt"

            with open(fname, "r") as f:
                lines = f.readlines()

            data.loc[alg, dataset, seed]["result"] = {
                "no": "incorrect",
                "yes": "correct",
                "almost": "almost",
                "incomp": "instability",
                "incomplete": "incomplete",
            }[lines[0].strip()]
            if len(lines) > 1:
                data.loc[alg, dataset, seed]["file"] = lines[1].strip()


num_correct = (data.result == "correct").groupby(level=[0, 1]).sum()
num_almost = (data.result == "almost").groupby(level=[0, 1]).sum()
num_incorrect = (data.result == "incorrect").groupby(level=[0, 1]).sum()
num_failed = (
    ((data.result == "instability") + (data.result == "incomplete"))
    .groupby(level=[0, 1])
    .sum()
)

num = num_correct + num_incorrect + num_almost
num_include_fails = num_correct + num_incorrect + num_almost + num_failed

percent_correct_include_fails = num_correct / num_include_fails * 100
percent_correct = num_correct / num * 100
percent_almost_include_fails = num_almost / num_include_fails * 100
percent_almost = num_almost / num * 100
# rediscovery_rate_with_fails = (num_correct / )
# rediscovery_rate = (num_correct / )
# approximate_rediscovery_rate = ((num_correct + num_almost)/ (num_correct + num_incorrect + num_almost))
# approximate_rediscovery_rate_with_fails = ((num_correct + num_almost)/ (num_correct + num_incorrect + num_almost + num_failed))
approximate_correct = num_correct + num_almost

# +
algorithm_columns = [
    "PySR",
    # "Eureqa",
    # "GPLearn",
    # "AI Feynman",
    "Operon",
    "DSR",
    # "PySINDy",
    "EQL",
    "QLattice",
    "SR-Transformer",
    # "GP-GOMEA",
    # r"SR Distillation$\ast$",
]

alg_mapping = {
    "PySR": "pysr",
    "Operon": "operon",
    "DSR": "uDSR",
    "EQL": "eql",
    "QLattice": "QLattice",
    "SR-Transformer": "E2ET",
}

num_algorithms = len(algorithm_columns)
num_columns = num_algorithms + 1

# Start table:


def generate_col_specification(frac, add_line=False):
    out = (
        r">{\hsize=" + str(frac) + r"\hsize\linewidth=\hsize\centering\arraybackslash}X"
    )
    if add_line:
        out = "|" + out
    return out


col_specification = ["l|"]
for alg in algorithm_columns:
    new_col = generate_col_specification(
        0.15, add_line=(alg == r"SR Distillation$\ast$")
    )
    col_specification.append(new_col)
col_specification = "".join(col_specification)

# fmt: off
output_table = [
    r"\begin{tabularx}{\textwidth}{" + col_specification + "}",
]

row = []
row.append(r"\toprule")
for col in algorithm_columns:
    if col == "PySR":
        col = r"\textbf{" + col + "}"
    col = r"\rotatebox[origin=c]{90}{\ " + col + r"\ }"
    row.append(col)
output_table.append(" & ".join(row))
# -

data = load()
problems = data["problems"]

skipped_problems = ["tully"]

# +
for i, (name, problem) in enumerate(problems.items()):
    key = problem["key"]
    if key in skipped_problems:
        continue

    row = [r"{ " + short_name[key] + "}"]
    if i == 0:
        row[0] = r"\midrule " + row[0]
    for alg in algorithm_columns:
        alg_key = alg_mapping[alg]
        # fmt: off
        n = num_correct.loc[alg_key, key]
        n_almost = num_almost.loc[alg_key, key]
        n_fail = num_failed.loc[alg_key, key]
        n_incorrect = num_incorrect.loc[alg_key, key]
        n_total = num_include_fails.loc[alg_key, key]
        color_scaling = n / n_total
        # cmap = plt.get_cmap("YlGn")
        # color = cmap(n / n_total)
        # Get rgb values
        # 37, 190, 88
        # to:
        # 123, 24, 24
        green = np.array([37, 190, 88]) / 255
        yellow = np.array([248, 255, 39]) / 255
        red = np.array([123, 24, 24]) / 255

        # At color_scaling = 0, we want red
        # At color_scaling = 0.5, we want yellow
        # At color_scaling = 1, we want green
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "mycmap", [red, yellow, green]
        )
        # rgb = rgb / 255.0
        color = cmap(color_scaling)
        rgb = np.array(color[:3])


        # Get xcolor string representation:
        color_str = r" \color[rgb]{" + ",".join([str(x) for x in rgb]) + "}"
        size_str = r" \Large "

        row.append(
                r"\begin{tabular}[x]{@{}c@{}}"
                + r" {$ " + color_str + size_str + r" \nicefrac{\bm{" + str(n) + r"}}{\bm{" + str(n_total) + r"}} $}\\"
                + r" {\color{gray} \tiny (" + ", ".join(map(str, [n, n_almost, n_fail, n_incorrect])) + ")}"
                + r"\end{tabular}"
            )
            # " $" + str(num_correct.loc[alg_key, key]) + r" \over " + str(num_include_fails.loc[alg_key, key]) + r"$ "
            # + r" {\color{gray}\small $" + str(num_correct.loc[alg_key, key]) + r" \over " + str(num.loc[alg_key, key]) + r"$}"
            # + r" {\color{gray}\small $" + str(approximate_correct.loc[alg_key, key]) + r" \over " + str(num_include_fails.loc[alg_key, key]) + r"$}"
            # + r" {\color{gray}\small $" + str(approximate_correct.loc[alg_key, key]) + r" \over " + str(num.loc[alg_key, key]) + r"$}"
            # fmt: on

    output_table.append(" & ".join(row))

output_table.append(r"\bottomrule\end{tabularx}")

with open(output_path / "results.tex", "w") as f:
    print("\\\\\n".join(output_table), file=f)
