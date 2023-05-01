rule compute_answer:
    input:
        "src/data/empirical_bench/equations.yml",
    output:
        "src/tex/output/table.tex",
    script:
        "src/scripts/equations_table.py"


rule compute_comparison:
    input:
        "src/data/pysr_comparison.xlsx",
    output:
        "src/tex/output/comparison_table.tex",
    script:
        "src/scripts/comparison.py"


rule compute_results:
    output:
        "src/tex/output/results.tex",
    script:
        "src/scripts/visualize_results.py"
