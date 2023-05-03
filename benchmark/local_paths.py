from pathlib import Path

this_dir = Path(__file__).parent

methods_path = this_dir / "official_competitors"
data_path = this_dir / ".." / "src" / "data" / "empirical_bench"
src_path = this_dir / ".." / "src"
results_path = this_dir / "results"