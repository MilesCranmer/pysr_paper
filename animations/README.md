# Animations

The code in this folder was used to create the animations in
[this tweet](https://twitter.com/MilesCranmer/status/1654169022852894721?s=20).

## How to run

First, install Python dependencies (including `manim` and `pysr`) in a new
virtual environment with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, install the Julia dependencies with:

```bash
python -c 'import pysr; pysr.install()'
```

then, initiate the local Julia environment with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Then, you can create the animations with, for example:

```bash
python -m manim -r 1980,1080 --fps=30 main.py [SCENE]
```

where `SCENE` is one of `TrigScene`, `ReluScene`, and `AbsScene`.

## Post-processing

The rest of the work was in Adobe After Effects to move
diagrams around and add the `glow` effect to make the look more striking.

The diagrams in the paper are found in [this file](https://github.com/MilesCranmer/pysr_paper/blob/main/src/static/pysr_diagram_v6.pdf)
which are raw vector diagrams and editable in Adobe Illustrator (all the figures are in a single PDF).
