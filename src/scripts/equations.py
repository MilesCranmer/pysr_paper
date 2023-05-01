# Read equations.yml and convert to table format.
import os

os.environ["MPLBACKEND"] = "agg"

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import matplotlib as mpl

from paths import data as data_path, figures as fig_path
from benchmark import load, gen_dataset

ebench_path = data_path / "empirical_bench"

# Matplotlib settings:
plt.rc("font", family="serif")

# plt.style.use("seaborn-darkgrid")
plt.style.use("bmh")

mp.dps = 50

skipped_problems = ["tully"]

def create_benchmark_figure(problems):
    xitems = 3
    yitems = 3

    fig, orig_ax = plt.subplots(yitems, xitems, figsize=(4 * xitems, 4 * yitems))

    # Don't plot in left lower and right lower:
    ax = [
        orig_ax[i, j]
        for i in range(yitems)
        for j in range(xitems)
        # if (not (i == 2 and j == 0)) and (not (i == 2 and j == 3))
    ]

    i = 0

    # Clear out (2, 0) and (2, 3) axes:
    # orig_ax[2, 0].axis("off")
    # orig_ax[2, 3].axis("off")

    for name, problem in problems.items():
        key = problem["key"]
        if "data_generator" not in problem and "data" not in problem:
            continue
        if key in skipped_problems:
            continue
        X, y = gen_dataset(data, key)

        ax[i].set_title(f'{name}\n${problem["equation"]["original"]}$')
        # color = "tab:blue"
        cmap = plt.get_cmap("viridis")
        normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
        if len(X.columns) > 1:
            color = X[X.columns[1]].values
            vmin = np.min(color)
            vmax = np.max(color)
        else:
            color = np.zeros_like(y)
            vmin = 0.1
            vmax = 1
        cscale = "linear"
        if (
            "plot" in problem
            and "cscale" in problem["plot"]
            and problem["plot"]["cscale"] == "log"
        ):
            # Define lognorm colormap:
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
            cscale = "log"
        else:
            # linear norm:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        im = ax[i].scatter(X[X.columns[0]], y, c=color, cmap=cmap, norm=norm)

        try:
            ax[i].set_xlabel(f"${X.columns[0]}$ ({problem['variables'][X.columns[0]]})")
        except KeyError:
            raise KeyError(f"{X.columns[0]} not in {problem['variables']}")
        ax[i].set_ylabel(
            f'${problem["equation"]["target"]}$ '
            f'({problem["variables"][problem["equation"]["target"]]})'
        )

        if "plot" in problem:
            if "xscale" in problem["plot"] and problem["plot"]["xscale"] == "log":
                ax[i].set_xscale("log")
            if "yscale" in problem["plot"] and problem["plot"]["yscale"] == "log":
                ax[i].set_yscale("log")
            if "xlim" in problem["plot"]:
                ax[i].set_xlim([float(l) for l in problem["plot"]["xlim"]])
            if "ylim" in problem["plot"]:
                ax[i].set_ylim([float(l) for l in problem["plot"]["ylim"]])

        # Add text saying "real data used" in upper left corner.
        text_string = []
        if "data" in problem:
            text_string.append("Original data used")
        if len(X.columns) > 1:
            # State what column is used for color, in text object:
            if cscale == "log":
                sci_notation_vmin = f"{vmin:.1e}".split("e")
                sci_notation_vmin = (
                    sci_notation_vmin[0] + "\\times 10^{" + sci_notation_vmin[1] + "}"
                )
                sci_notation_vmax = f"{vmax:.1e}".split("e")
                sci_notation_vmax = (
                    sci_notation_vmax[0] + "\\times 10^{" + sci_notation_vmax[1] + "}"
                )
            else:
                sci_notation_vmin = f"{vmin:.1f}"
                sci_notation_vmax = f"{vmax:.1f}"
            text_string.append(
                f"${X.columns[1]}$ shown in color $\in [{sci_notation_vmin}, {sci_notation_vmax}]$"
            )

        text_string = "\n".join(text_string)

        if len(text_string) > 0:
            bbox = dict(facecolor="white", edgecolor="black", pad=2)
            # Make bbox transparent:
            bbox["alpha"] = 0.5
            ax[i].text(
                0.02,
                0.98,
                text_string,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax[i].transAxes,
                zorder=int(1e9),
                bbox=bbox,
            )

        i += 1

    plt.tight_layout()
    plt.savefig(fig_path / "benchmark_data.pdf")


if __name__ == "__main__":
    data = load()
    problems = data["problems"]
    create_benchmark_figure(problems)
