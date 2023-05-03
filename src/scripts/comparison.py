# This file generates a LaTeX table to compare PySR with other libraries.
from re import sub
import pandas as pd
from paths import output as output_path, data as data_path

filename = "pysr_comparison.xlsx"
sheet_name = "comparison"

data = pd.read_excel(data_path / filename, sheet_name=sheet_name)

# Keys:
# Index(['Order', 'Importance for Science (1-3)', 'Category', 'Subcategory',
#        'PySR', 'Eureqa', 'GPLearn', 'glyph/DEAP', 'PySINDy', 'EQL',
#        'AI Feynman', 'Operon', 'DSR', 'SR-Transformer', 'GP-GOMEA', 'QLattice',
#        'uDSR', '<any package> used with Cranmer+2020', 'Notes'],
#       dtype='object')

columns = data.columns
non_algorithm_columns = [
    "Order",
    "Importance for Science (1-3)",
    "Category",
    "Subcategory",
    "Notes",
]
# algorithm_columns = [col for col in columns if col not in non_algorithm_columns]
algorithm_columns = [
    "PySR",
    "Eureqa",
    "GPLearn",
    "AI Feynman",
    "Operon",
    "DSR",
    "PySINDy",
    "EQL",
    "QLattice",
    "SR-Transformer",
    "GP-GOMEA",
    r"SR Distillation$\ast$",
]

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


col_specification = ["l", "l|"]
for alg in algorithm_columns:
    new_col = generate_col_specification(
        0.1, add_line=(alg == r"SR Distillation$\ast$")
    )
    col_specification.append(new_col)
col_specification = "".join(col_specification)

# fmt: off
output_table = [
    r"\begin{tabularx}{\linewidth}{" + col_specification + "}",
    r"\rowcolor{gray!50}",
    r"\toprule",
]
# fmt: on

text_based_subcategories = [
    "Method Family",
    "Programming Language",
    "Expressivity",
]

# Header:
row = []
row.append("")
row.append("")
for col in algorithm_columns:
    if col == "PySR":
        col = r"\textbf{" + col + "}"
    col = r"\rotatebox[origin=c]{90}{\ " + col + r"\ }"
    row.append(col)
output_table.append(" & ".join(row) + r"\\")

# Move all text to a footnote of the table:
footnotes = {}
# fmt: off
footnote_counter = iter([
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
])
# fmt: on
replacement_table = {
    "nan": "-",
    "1": r"${\color{green}\checkmark}$",
    "0": r"${\color{red}\cross}$",
}

cur_category = ""
min_color = 0.8
max_color = 1.0  # 0.97
cur_color = max_color
min_color_category = 0.85
max_color_category = 0.95
cur_color_category = min_color_category
n_cur_category = 0

# if subcategory_name in text_based_subcategories:
#     continue
# Filter out text_based_subcategories:
filtered_data = data[~data["Subcategory"].isin(text_based_subcategories)]
# Remove categories with importance of 0:
filtered_data = filtered_data[filtered_data["Importance for Science (1-3)"] != 0]

# Recompute indices:
filtered_data = filtered_data.reset_index(drop=True)

# Entries:
for i, category in filtered_data.iterrows():
    row = []

    category_name = category["Category"]
    subcategory_name = category["Subcategory"]

    if category_name != cur_category:
        cur_category = category_name
        n_cur_category = 0
        cur_color_category = {
            min_color_category: max_color_category,
            max_color_category: min_color_category,
        }[cur_color_category]

    n_cur_category += 1

    # Check if category changes next row:
    last_row_of_category = (
        i + 1 == len(filtered_data)
        or filtered_data.iloc[i + 1]["Category"] != category_name
    )

    importance = category["Importance for Science (1-3)"]
    # adjusted_cur_color = cur_color_category * cur_color
    adjusted_cur_color = cur_color
    for alg in algorithm_columns:
        entry = str(category[alg])

        if entry in replacement_table.keys():
            # Replace with check/x:
            entry = replacement_table[entry]
        elif subcategory_name == "Citation":
            # Wrap with citation if needed:
            if entry != "[self]":
                entry = r"\cite{" + entry + "}"
        elif subcategory_name == "Code":
            if entry.startswith("https://github"):
                # Wrap link with "GitHub" hyperlink:
                entry = r"\href{" + entry + r"}{\faUnlock}"
            else:
                # Leave x:
                # entry = replacement_table["0"]
                entry = r"\faLock"
        elif len(entry) > 3:
            # First, check if the entry is the same as any other footnotes:
            i_footnote = None
            if entry not in footnotes.values():
                # If text, replace with a footnote:
                i_footnote = next(footnote_counter)
                footnotes[i_footnote] = entry
            else:
                # Find the corresponding footnote:
                for _i_footnote, _footnote in footnotes.items():
                    if entry == _footnote:
                        i_footnote = _i_footnote
                        break

            entry = r"$\ast$" + i_footnote

        entry = r"\cellcolor[gray]{" + str(adjusted_cur_color) + "}" + entry
        # Include:
        row.append(entry)

    row = (
        r"\cellcolor[gray]{"
        + str(adjusted_cur_color)
        + "}"
        + r"{"
        + subcategory_name
        + "} & "
        + " & ".join(row)
        + r"\\"
    )

    if last_row_of_category:
        # Create multirow and stretch up:
        row = (
            r"\multirow{-"
            + str(n_cur_category)
            + r"}{*}{\cellcolor[gray]{"
            + str(cur_color_category)
            + r"}\textbf{"
            + category_name
            + r"}} & "
            + row
        )
    else:
        # Otherwise, color normally:
        row = r"\cellcolor[gray]{" + str(cur_color_category) + "}" + "& " + row

    cur_color = {
        min_color: max_color,
        max_color: min_color,
    }[cur_color]

    output_table.append(row)

output_table.append(r"\bottomrule")
add_footnotes = True
# Add footnotes:
if add_footnotes:
    output_table.append(r"\end{tabularx}")
    output_table.append(r"\begin{tabularx}{\linewidth}{lX}")
    output_table.append(r"\toprule")
    # Custom note about SR Distillation:
    output_table.append(
        r"\extranotes"
        # r"$\ast$"
        # + r" & "
        # + r"\srdistillation."
        # + r"\\"
    )
    for footnote in footnotes.keys():
        output_table.append(r"$\ast$" + footnote + r" & " + footnotes[footnote] + r"\\")
    output_table.append(r"\bottomrule")

# End table:
output_table.append(r"\end{tabularx}")

output_table = "\n".join(output_table)
# Replace
#   r"SR Distillation$\ast$"
# with
#   "Symbolic\nDistillation" + r"$\ast$":

# output_table = output_table.replace(
#     r"SR Distillation$\ast$", "Symbolic Distillation" + r"$\ast$")
output_table = output_table.replace(
    r"SR Distillation$\ast$", "\\begin{tabular}{@{}c@{}}Symbolic \\\\ Distillation" + r"$\ast$" + "\\end{tabular}")

output_table = output_table.replace(
    "virgolinImprovingModelbasedGenetic2021a", "virgolinImprovingModelbasedGenetic2021")

print(output_table, file=open(output_path / "comparison_table.tex", "w"))
