from equations import load
from paths import output as output_path

skipped_problems = ["tully"]
# ^ Tully is not even the correct expression!


def create_table(problems):
    output_str = []

    for i, (name, problem) in enumerate(problems.items()):
        key = problem["key"]
        if key in skipped_problems:
            continue
        if "data_generator" not in problem and "data" not in problem:
            continue
        # X, y = gen_dataset(data, key)
        if "citations" in problem and "original" in problem["citations"] and problem["citations"]["original"]:
            citation_info = r"\cite{" + problem["citations"]["original"] + "}"
        else:
            citation_info = ""

        output_str.append(
            " & ".join(
                [f"{name}", f'${problem["equation"]["original"]}$', citation_info]
            )
            + r"\\"
        )

    # Clean up last line:
    output_str[-1] = output_str[-1][:-2]

    with open(output_path / "table.tex", "w") as f:
        f.write("\n".join(output_str))

    print("\n".join(output_str))


if __name__ == "__main__":
    data = load()
    problems = data["problems"]
    create_table(problems)
