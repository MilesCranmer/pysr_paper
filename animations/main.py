from manim import *
import numpy as np
from utils import jl
from utils import sr
from utils import div
from utils import SRScene
from utils import Event
from utils import gen_transition_sequence


class TrigScene(SRScene):
    """
    Truth: (x + sin(y + 1.3)) / 0.5
    Steps:
        x + y
        x + cos(y)
        x + cos(y + 1.5)
        x + sin(y + 1.5)
        x + sin(y + {opt})
        (x + sin(y + [opt])) / 0.8
        (x + sin(y + {opt})) / {opt}
    """

    def construct(self):
        x, y, z = [sr.Node(jl.Float64, feature=i) for i in [1, 2, 3]]
        options = jl.eval(
            """
            options = Options(binary_operators=[+, -, *, /], unary_operators=[cos, sin])
            @extend_operators options
            options
        """
        )
        true_f_str = "2 * (x + sin(y + 1.3))"
        true_f = lambda x, y, z: (x + np.sin(y + 1.3)) * 2
        feature_map = ["x", "y"]

        events = [
            Event(jl.plus(x, y)),
            Event(
                jl.plus(x, jl.cos(y)),
                mutation="Add node",
                highlight=(["r"], ["r", "r.l"]),
            ),
            Event(
                jl.plus(x, jl.cos(jl.plus(y, 1.5))),
                mutation="Add node",
                highlight=(["r.l"], ["r.l", "r.l.l", "r.l.r"]),
            ),
            Event(
                jl.plus(x, jl.sin(jl.plus(y, 1.5))),
                mutation="Mutate node",
                highlight=(["r"], ["r"]),
            ),
            Event(
                type="optimization",
                tree_gen=lambda c: jl.plus(x, jl.sin(jl.plus(y, c[0]))),
                get_const_history=lambda t: jl.eval(
                    f"""
                    get_optimization_history(;
                        true_f=(x, y, z) -> {true_f_str},
                        test_f=(x, y, z; c) -> plus(x, sin(plus(y, c[1]))),
                        c0=[1.5],
                        c_bounds=[(0.5, 2.0)],
                        iterations=10
                    )
                """
                ),
                highlight=["r.l.r"],
            ),
            Event(
                tree=(lambda ltree: div(sr.copy_node(ltree), 0.8),),
                mutation="Prepend node",
                highlight=(None, ["", "r"]),
            ),
            Event(
                type="optimization",
                tree_gen=lambda c: div(jl.plus(x, jl.sin(jl.plus(y, c[0]))), c[1]),
                get_const_history=lambda t: jl.eval(
                    f"""
                        get_optimization_history(;
                            true_f=(x, y, z) -> {true_f_str},
                            test_f=(x, y, z; c) -> div(plus(x, sin(plus(y, c[1]))), c[2]),
                            c0=[{t.l.r.l.r.val}, 0.8],
                            c_bounds=[(0.5, 2.0), (0.2, 1.5)],
                            iterations=20
                        )
                    """
                ),
                highlight=["l.r.l.r", "r"],
            ),
        ]
        transitions = gen_transition_sequence(events)

        super().construct(
            transitions=transitions,
            true_f=true_f,
            feature_map=feature_map,
            options=options,
        )


class ReluScene(SRScene):
    """
    Truth: 0.8 * relu(z + 2.5 * x) + exp(-0.1 * y)
    Steps:
        1.2 * relu(x)
        1.2 * relu(1.8 * x)
        1.2 * relu(z + 1.8 * x)
        1.2 * relu(z + 1.8 * x + x)
        1.2 * relu(z + 2.8 * x)
        {opt} * relu(z + {opt} * x)
        [opt] * relu(z + [opt] * x) + exp(-0.08)
        [opt] * relu(z + [opt] * x) + exp(-0.08 * y)
        {opt} * relu(z + {opt} * x) + exp({opt} * y)
    """

    def construct(self):
        x, y, z = [sr.Node(jl.Float64, feature=i) for i in [1, 2, 3]]
        nprelu = lambda x: np.where(x > 0, x, 0)
        true_f_str = "0.8 * relu(z + 2.5 * x) + exp(-0.1 * y)"
        true_f = lambda x, y, z: 0.8 * nprelu(z + 2.5 * x) + np.exp(-0.1 * y)
        options = jl.eval(
            """
            options = Options(binary_operators=[+, -, *, /], unary_operators=[cos, sin, relu, exp])
            @extend_operators options
            options
        """
        )
        feature_map = ["x", "y", "z"]
        events = [
            Event(sr.mult(1.2, sr.relu(x))),
            Event(
                sr.mult(1.2, sr.relu(sr.mult(1.8, x))),
                mutation="Add node",
                highlight=(["r.l"], ["r.l", "r.l.l", "r.l.r"]),
            ),
            Event(
                sr.mult(1.2, sr.relu(sr.plus(z, sr.mult(1.8, x)))),  # r.l is the plus
                mutation="Insert node",
                highlight=(["r.l"], ["r.l", "r.l.l", "r.l.r"]),
            ),
            Event(
                sr.mult(1.2, sr.relu(sr.plus(z, sr.plus(sr.mult(1.8, x), x)))),
                mutation="Insert node",
                highlight=(["r.l.r"], ["r.l.r", "r.l.r.l", "r.l.r.r"]),
            ),
            Event(
                sr.mult(1.2, sr.relu(sr.plus(z, sr.mult(2.8, x)))),
                mutation="Simplify",
                highlight=(
                    ["r.l.r.l.l", "r.l.r.l.r", "r.l.r.l", "r.l.r", "r.l.r.r"],
                    ["r.l.r.l", "r.l.r.r", "r.l.r"],
                ),
            ),
            Event(
                type="optimization",
                tree_gen=lambda c: sr.mult(c[0], sr.relu(sr.plus(z, sr.mult(c[1], x)))),
                get_const_history=lambda t: jl.eval(
                    f"""
                        get_optimization_history(;
                            true_f=(x, y, z) -> {true_f_str},
                            test_f=(x, y, z; c) -> mult(c[1], relu(plus(z, mult(c[2], x)))),
                            c0=[1.2, 2.8],
                            c_bounds=[(0.0, 10.0), (0.0, 10.0)],
                            iterations=10
                        )
                    """
                ),
                highlight=["l", "r.l.r.l"],
            ),
            # {opt} * relu(z + {opt} * x)
            # [opt] * relu(z + [opt] * x) + exp(-0.08)
            # [opt] * relu(z + [opt] * x) + exp(-0.08 * y)
            # {opt} * relu(z + {opt} * x) + exp({opt} * y)
            Event(
                (
                    lambda ltree: jl.plus(
                        jl.mult(
                            ltree.l.val,
                            jl.relu(jl.plus(z, jl.mult(ltree.r.l.r.l.val, x))),
                        ),
                        sr.Node(4, sr.Node(jl.Float64, val=-0.08)),
                    ),
                ),
                mutation="Prepend node",
                highlight=(None, ["", "r", "r.l"]),
            ),
            Event(
                (
                    lambda ltree: jl.plus(
                        jl.mult(
                            ltree.l.l.val,
                            jl.relu(jl.plus(z, jl.mult(ltree.l.r.l.r.l.val, x))),
                        ),
                        jl.exp(
                            jl.mult(-0.08, y)
                        ),  # r is exp. r.l is const and then is *. r.l.l is moved const, r.l.r is y.
                    ),
                ),
                mutation="Add node",
                highlight=(["r.l"], ["r.l", "r.l.r"]),
            ),
            Event(
                type="optimization",
                tree_gen=lambda c: jl.plus(
                    jl.mult(c[0], jl.relu(jl.plus(z, jl.mult(c[1], x)))),
                    jl.exp(jl.mult(c[2], y)),
                ),
                get_const_history=lambda t: jl.eval(
                    f"""
                        get_optimization_history(;
                            true_f=(x, y, z) -> {true_f_str},
                            test_f=(x, y, z; c) -> plus(mult(c[1], relu(plus(z, mult(c[2], x)))), exp(mult(c[3], y))),
                            c0=[{t.l.l.val}, {t.l.r.l.r.l.val}, {t.r.l.l.val}],
                            c_bounds=[(0.0, 10.0), (0.0, 10.0), (-0.2, -0.0)],
                            iterations=10,
                            alg=BFGS(),
                        )
                    """
                ),
                highlight=["l.l", "l.r.l.r.l", "r.l.l"],
            ),
        ]
        transitions = gen_transition_sequence(events)

        super().construct(
            transitions=transitions,
            true_f=true_f,
            feature_map=feature_map,
            options=options,
        )


class AbsScene(SRScene):
    """
    Truth: 1.2 * abs(x^2 - y) + 0.5 * cos(y) * sin(z^2)
    Steps:
        x^2 - y
        abs(x^2 - y)
        1.7 * abs(x^2 - y)
        {opt} * abs(x^2 - y)
        [opt] * abs(x^2 - y) + cos(y)
        [opt] * abs(x^2 - y) + 0.2 * cos(y) * z
        {opt} * abs(x^2 - y) + {opt} * cos(y) * z
        [opt] * abs(x^2 - y) + [opt] * cos(y) * sin(z)
        [opt] * abs(x^2 - y) + [opt] * cos(y) * sin(z^2)
        {opt} * abs(x^2 - y) + {opt} * cos(y) * sin(z^2)
    """

    def construct(self):
        x, y, z = [sr.Node(jl.Float64, feature=i) for i in [1, 2, 3]]
        options = jl.eval(
            """
            options = Options(binary_operators=[+, -, *, /], unary_operators=[cos, sin, abs])
            @extend_operators options
            options
        """
        )
        true_f_str = "1.2 * abs(x^2 - y) + 0.5 * cos(y) * sin(z^2)"
        true_f = lambda x, y, z: 1.2 * np.abs(x**2 - y) + 0.5 * np.cos(y) * np.sin(
            z**2
        )
        feature_map = ["x", "y", "z"]

        events = [
            Event(jl.sub(jl.mult(x, x), y)),
            Event(
                jl.abs(jl.sub(jl.mult(x, x), y)),
                mutation="Add node",
                highlight=(["r"], ["r"]),
            ),
            Event(
                sr.mult(1.7, jl.abs(jl.sub(jl.mult(x, x), y))),
                mutation="Add node",
                highlight=(["r"], ["r"]),
            ),
            Event(
                type="optimization",
                tree_gen=lambda c: sr.mult(c[0], jl.abs(jl.sub(jl.mult(x, x), y))),
                get_const_history=lambda t: jl.eval(
                    f"""
                        get_optimization_history(;
                            true_f=(x, y, z) -> {true_f_str},
                            test_f=(x, y, z; c) -> mult(c[1], abs(sub(square(x), y))),
                            c0=[1.7],
                            c_bounds=[(0.5, 2.0)],
                            iterations=10
                        )
                    """
                ),
                highlight=["l"],
            ),
            Event(
                (
                    lambda ltree: jl.plus(
                        sr.mult(ltree.l.val, jl.abs(jl.sub(jl.mult(x, x), y))),
                        jl.cos(y),
                    ),
                ),
                mutation="Prepend node",
                highlight=(None, ["", "r"]),
            ),
            Event(
                (
                    lambda ltree: jl.plus(
                        sr.mult(ltree.l.l.val, jl.abs(jl.sub(jl.mult(x, x), y))),
                        sr.mult(0.7, sr.mult(jl.cos(y), z)),
                    ),
                ),
                mutation="Insert node",
                highlight=(["r", "r.l"], ["r", "r.l", "r.r"]),
            ),
            Event(
                type="optimization",
                tree_gen=lambda c: jl.plus(
                    sr.mult(c[0], jl.abs(jl.sub(jl.mult(x, x), y))),
                    sr.mult(c[1], sr.mult(jl.cos(y), z)),
                ),
                get_const_history=lambda t: jl.eval(
                    f"""
                        get_optimization_history(;
                            true_f=(x, y, z) -> {true_f_str},
                            test_f=(x, y, z; c) -> plus(mult(c[1], abs(sub(square(x), y))), mult(c[2], mult(cos(y), z))),
                            c0=[{t.l.l.val}, {t.r.l.val}],
                            c_bounds=[(0.5, 2.0), (0.3, 1.0)],
                            iterations=10
                        )
                    """
                ),
                highlight=["l.l", "r.l"],
            ),
            Event(
                (
                    lambda ltree: jl.plus(
                        sr.mult(ltree.l.l.val, jl.abs(jl.sub(jl.mult(x, x), y))),
                        sr.mult(ltree.r.l.val, sr.mult(jl.cos(y), jl.sin(z))),
                    ),
                ),
                mutation="Add node",
                highlight=(["r.r"], ["r.r"]),
            ),
            Event(
                (
                    lambda ltree: jl.plus(
                        sr.mult(
                            ltree.l.l.val,
                            jl.abs(jl.sub(jl.mult(x, x), y)),
                        ),
                        sr.mult(
                            ltree.r.l.val,
                            sr.mult(jl.cos(y), jl.sin(jl.square(z))),
                        ),
                    ),
                ),
                mutation="Add node",
                highlight=(["r.r.r"], ["r.r.r"]),
            ),
            Event(
                type="optimization",
                tree_gen=lambda c: jl.plus(
                    sr.mult(c[0], jl.abs(jl.sub(jl.mult(x, x), y))),
                    sr.mult(c[1], sr.mult(jl.cos(y), jl.sin(jl.square(z)))),
                ),
                get_const_history=lambda t: jl.eval(
                    f"""
                        get_optimization_history(;
                            true_f=(x, y, z) -> {true_f_str},
                            test_f=(x, y, z; c) -> plus(mult(c[1], abs(sub(square(x), y))), mult(c[2], mult(cos(y), sin(square(z))))),
                            c0=[{t.l.l.val}, {t.r.l.val}],
                            c_bounds=[(1.0, 2.0), (0.2, 1.0)],
                            iterations=20
                        )
                    """
                ),
                highlight=["l.l", "r.l"],
            ),
        ]

        transitions = gen_transition_sequence(events)

        super().construct(
            transitions=transitions,
            true_f=true_f,
            feature_map=feature_map,
            options=options,
        )
