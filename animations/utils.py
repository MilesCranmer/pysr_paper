import re
from manim import *
from pysr.julia_helpers import init_julia
from functools import partial, reduce
from collections import namedtuple
import random
import string

jl = init_julia(julia_kwargs={"threads": "auto", "optimize": 3})
from julia import Pkg

Pkg.activate(".")

jl.include("utils.jl")
jl.include("optimization_example.jl")
from julia import SymbolicRegression as sr

config["max_files_cached"] = 10000

BACKGROUND_COLOR = "#15191F"
# Bluish green: "#3E746F"
DESCRIPTION_COLOR = WHITE

GLOBAL_SCALE = 0.5
TEXT_SCALE = 0.9
DEFAULT_STROKE = 6
DESCRIPTION_SCALE = TEXT_SCALE * 2.5

TEXT_Z_INDEX = 3
CIRCLE_Z_INDEX = 2
LINE_Z_INDEX = 1
PLOT_Z_INDEX = 4

START_POS = ORIGIN + 2 * UP + 3 * LEFT

# Positionings:
PLOT_OFFSET = 13 * RIGHT + 5 * DOWN
EQUATION_OFFSET = 11 * RIGHT + 2 * UP
DESCRIPTION_OFFSET = 3 * LEFT + 3 * UP
LEGEND_OFFSET = 4 * DOWN + 3 * RIGHT

# Timings:
DEFAULT_TRANSITION_TIME = 2
PERTURBATION_TIME = DEFAULT_TRANSITION_TIME
COLOR_FADE_IN_MULTIPLIER = 1 / 8
SHAPE_TRANSFORMATION_MULTIPLIER = 1 / 2
COLOR_FADE_OUT_MULTIPLIER = 1 / 8
INTERMISSION_MULTIPLIER = 1 / 4

PERTURBATION_OPACITY = 0.2


# TODO: Ensure these are the same with optimization_example.jl!
get_y = lambda x: 2 * x + 0.5
get_z = lambda x: -1.5 * x + 0.2

div = jl.eval("(/)")
jl.eval("div(x, y) = x / y")


def to_string(tree, options, bracketed=True, full_precision=False, feature_map=None):
    if tree.degree > 0:
        op = str(jl.just_op_to_string(tree, options))
        children = [tree.l, tree.r] if tree.degree == 2 else [tree.l]
        if op in ["+", "-", "*", "^", "/", r"\times"]:
            if op == "/":
                s = (
                    r"{"
                    + to_string(
                        children[0],
                        options=options,
                        bracketed=True,
                        full_precision=full_precision,
                        feature_map=feature_map,
                    )
                    + r" \over "
                    + to_string(
                        children[1],
                        options=options,
                        bracketed=True,
                        full_precision=full_precision,
                        feature_map=feature_map,
                    )
                    + "}"
                )
                if bracketed:
                    return s
                return r"\left(" + f"{s}" + r" \right)"
            else:
                if op == r"\times":
                    op = r"\cdot"
                s = f" {op} ".join(
                    [
                        to_string(
                            child,
                            options=options,
                            bracketed=False,
                            full_precision=full_precision,
                            feature_map=feature_map,
                        )
                        for child in children
                    ]
                )
                if bracketed:
                    return s
                return r"\left(" + f"{s}" + r" \right)"
        elif r"\lvert" in op:
            return (
                r"\left\lvert"
                + f"{', '.join([to_string(child, options=options, bracketed=True, full_precision=full_precision, feature_map=feature_map) for child in children])}"
                + r"\right\rvert"
            )
        else:
            return (
                f"{op}"
                + r"\left("
                + f"{', '.join([to_string(child, options=options, bracketed=True, full_precision=full_precision, feature_map=feature_map) for child in children])}"
                + r" \right)"
            )
    else:
        if tree.constant:
            if full_precision:
                return str(tree.val)
            else:
                return f"{tree.val:.2f}"
        else:
            return feature_map[tree.feature - 1]


class Expression:
    def __init__(self, tree, pos=START_POS):
        self.tree = sr.copy_node(tree)
        self.pos = pos


class ExpressionTransition:
    def __init__(
        self,
        tree1,
        tree2,
        duration=DEFAULT_TRANSITION_TIME,
        transition_type=TransformMatchingShapes,
        mutation=None,
        highlight=None,
        perturb=False,
        event=None,
    ):
        assert not tree1 is None
        assert not tree2 is None
        if not isinstance(tree1, Expression):
            tree1 = Expression(tree1)
        if not isinstance(tree2, Expression):
            tree2 = Expression(tree2)
        self.tree1 = tree1
        self.tree2 = tree2
        self.duration = duration
        self.transition_type = transition_type
        self.mutation = mutation
        if highlight is not None and len(highlight) == 2:
            highlight = [None, *highlight, None]
        elif highlight is None:
            highlight = [None, None, None, None]
        assert len(highlight) == 4
        self.highlight = highlight
        self.perturb = perturb
        self.event = event


def jlset(obj, field, val):
    jl.setproperty_b(obj, field, val)


def matching_func(m1, m2):
    if type(m1) != type(m2):
        return False, np.inf
    if isinstance(m1, VGroup):
        # Check if it's a MathTex + Circle. Otherwise
        # just return.
        if len(m1) != 2 or len(m2) != 2:
            return False, np.inf
        tex1, circle1 = m1
        tex2, circle2 = m2
        if (
            not isinstance(tex1, MathTex)
            or not isinstance(circle1, Circle)
            or not isinstance(tex2, MathTex)
            or not isinstance(circle2, Circle)
        ):
            return False, np.inf

        # Check if the text and position is the same.
        if tex1.get_tex_string() != tex2.get_tex_string():
            return False, np.inf
        return True, np.linalg.norm(tex1.get_center() - tex2.get_center())
    if isinstance(m1, Line):
        # Check if starting and ending points are the same.
        if not (m1.get_start() == m2.get_start()).all():
            return False, np.inf
        if not (m1.get_end() == m2.get_end()).all():
            return False, np.inf
        return True, np.inf
    return False, np.inf


def smart_transform(vgroup1: VGroup, vgroup2: VGroup, transition_type):
    unmatched_1 = []
    unmatched_2 = []
    matches = {}

    for m1 in vgroup1:
        tmp_matches = []
        for m2 in vgroup2:
            if m2 in matches.values():
                continue
            match, distance = matching_func(m1, m2)
            if match:
                tmp_matches.append((m2, distance))

        if len(tmp_matches) == 0:
            unmatched_1.append(m1)
        elif len(tmp_matches) == 1:
            for m2, _ in tmp_matches:
                matches[m1] = m2
        else:
            # Multiple matches, so pick the best one.
            best_match = None
            best_distance = np.inf
            for m2, distance in tmp_matches:
                if best_match is None or distance < best_distance:
                    best_match = m2
                    best_distance = distance
            matches[m1] = best_match

    for m2 in vgroup2:
        if m2 not in matches.values():
            unmatched_2.append(m2)

    # Now, we create animations between the matched objects.
    animations = []
    for m1, m2 in matches.items():
        animations.append(transition_type(m1, m2))
    animations.append(transition_type(VGroup(*unmatched_1), VGroup(*unmatched_2)))
    return animations


class SRScene(Scene):
    def construct(self, *, transitions, true_f, feature_map, options):
        self.true_f = true_f
        self.feature_map = feature_map
        self.options = options
        self.CACHE = {}
        # set background color
        self.set_background()

        # Plot truth and prediction and two curves.
        # plot_x vs truth:

        for i, transition in enumerate(transitions):
            if isinstance(transition, Expression):
                l, cn = self.create_tree(
                    transition.tree,
                    scale=GLOBAL_SCALE,
                    pos=transition.pos,
                    z_index=0,
                    highlight=None,
                )
                tree_latex = self.create_equation(transition.tree)
                plot = self.create_plot(transition.tree, color=WHITE)
                self.play(Write(l), Write(cn), Write(tree_latex), Write(plot))
                self.wait(DEFAULT_TRANSITION_TIME * INTERMISSION_MULTIPLIER)
            elif isinstance(transition, ExpressionTransition):
                l1_pre_color, cn1_pre_color = self.create_tree(
                    transition.tree1.tree,
                    scale=GLOBAL_SCALE,
                    pos=transition.tree1.pos,
                    highlight=transition.highlight[0],
                    z_index=0,
                )
                l1, cn1 = self.create_tree(
                    transition.tree1.tree,
                    scale=GLOBAL_SCALE,
                    pos=transition.tree1.pos,
                    highlight=transition.highlight[1],
                    z_index=0,
                )
                p1 = self.create_plot(transition.tree1.tree, color=WHITE)
                p2 = self.create_plot(transition.tree2.tree, color=WHITE)
                l2, cn2 = self.create_tree(
                    transition.tree2.tree,
                    scale=GLOBAL_SCALE,
                    pos=transition.tree2.pos,
                    highlight=transition.highlight[2],
                    z_index=0,
                )
                l2_post_color, cn2_post_color = self.create_tree(
                    transition.tree2.tree,
                    scale=GLOBAL_SCALE,
                    pos=transition.tree2.pos,
                    highlight=transition.highlight[3],
                    z_index=0,
                )

                second_phase = []
                frozen_objs = []
                perturbations = None
                n_new_perturbations = 15
                n_perturbations = n_new_perturbations + 1
                if transition.perturb:
                    x_range = [-5, 5]
                    test_x = np.linspace(*x_range, 1000)
                    test_y = get_y(test_x)
                    test_z = get_z(test_x)
                    test_X = np.array([test_x, test_y, test_z])
                    perturbed_trees = [
                        jl.create_perturbed_tree(
                            transition.tree1.tree,
                            options=self.options,
                            nfeatures=len(self.feature_map),
                            X=test_X,
                            max_mutations=2,
                        )
                        for _ in range(n_new_perturbations)
                    ]
                    perturbed_trees.append(transition.tree2.tree)
                    print("Created perturbed trees.")
                    hex_to_rgb = lambda s: np.array(
                        [int(s[i : i + 2], 16) for i in (0, 2, 4)]
                    )
                    rgb_to_hex = lambda rgb: "#%02x%02x%02x" % tuple(
                        np.floor(rgb).astype(int)
                    )
                    get_transparent_color = lambda opacity: rgb_to_hex(
                        hex_to_rgb(BACKGROUND_COLOR[1:]) * (1 - opacity)
                        + np.array([255, 255, 255]) * opacity
                    )

                    perturbations = [
                        VGroup(
                            *[
                                *[
                                    el.set_opacity(PERTURBATION_OPACITY)
                                    for el in self.create_tree(
                                        tree,
                                        scale=GLOBAL_SCALE,
                                        pos=transition.tree1.pos,
                                        highlight=None,
                                        z_index=-10 - 10 * i,
                                        default_stroke_width=DEFAULT_STROKE * 0.9,
                                        extra_hash="perturbation",
                                    )
                                ],
                                self.create_plot(
                                    tree,
                                    color=get_transparent_color(PERTURBATION_OPACITY),
                                    z_index=-10 - 10 * i,
                                    extra_hash="perturbation",
                                ),
                            ]
                        )
                        for i, tree in enumerate(perturbed_trees)
                    ]

                    ## Perturbation phase:
                    pert_animations = []
                    # Construct a Succession for each perturbation.
                    # Want the total time to be `PERTURBATION_TIME`.
                    # We want each tree to come it at a different time (i.e., Wait()), FadeIn,
                    # Wait, then fadeout, then Wait. At any time, we want
                    # for n trees to be visible.
                    # Say that fade_in_time = PERTURBATION_TIME / 8
                    # If fade_in_time = PERTURBATION_TIME / 2, then
                    # n_trees_visible = n_perturbations.
                    # This scales linearly with fade_in_time. Thus,
                    # n_trees_visible = n_perturbations * (fade_in_time / (PERTURBATION_TIME / 2))
                    # Invert this:
                    n_trees_visible = 4
                    fade_in_time = (
                        n_trees_visible / n_perturbations * (PERTURBATION_TIME / 2)
                    )
                    for ip, obj in enumerate(perturbations):
                        animation_parts = []
                        animation_parts.append(
                            FadeIn(obj, run_time=fade_in_time, rate_func=linear)
                        )
                        if ip == len(perturbations) - 1:
                            # Fade out the last one when we are doing the second phase (transforming equations)
                            animation_parts.append(Wait(fade_in_time))
                            second_phase.append(FadeOut(obj, rate_func=linear))
                            frozen_objs.append(obj)
                        else:
                            animation_parts.append(
                                FadeOut(obj, run_time=fade_in_time, rate_func=linear)
                            )

                        pert_animations.append(Succession(*animation_parts))

                    self.play(
                        AnimationGroup(*pert_animations, lag_ratio=1 / n_trees_visible),
                        run_time=PERTURBATION_TIME,
                    )

                ## Write description:
                tree_latex1 = self.create_equation(transition.tree1.tree)
                tree_latex2 = self.create_equation(transition.tree2.tree)
                descriptor_latex = self.create_description(transition.mutation)

                is_last = i == len(transitions) - 1
                if (
                    not hasattr(transitions[i - 1], "mutation")
                    or transition.event != transitions[i - 1].event
                ):
                    anis = [
                        Write(descriptor_latex),
                        *[
                            TransformMatchingShapes(obj, obj, run_time=0.0001)
                            for obj in frozen_objs
                        ],
                    ]
                    self.play(
                        *anis,
                        run_time=0.0001,
                    )
                transform = transition.transition_type

                ## First phase:
                if l1 != l1_pre_color or cn1 != cn1_pre_color:
                    anis = [
                        transform(l1_pre_color, l1),
                        transform(cn1_pre_color, cn1),
                        *[
                            TransformMatchingShapes(obj, obj, run_time=0.0001)
                            for obj in frozen_objs
                        ],
                    ]
                    self.play(
                        *anis,
                        path_arc=0 * DEGREES,
                        rate_func=linear,
                        run_time=transition.duration * COLOR_FADE_IN_MULTIPLIER,
                    )

                ## Second phase:
                second_phase = [
                    TransformMatchingShapes(p1, p2, path_arc=90 * DEGREES),
                    *smart_transform(l1, l2, transition_type=transform),
                    *smart_transform(cn1, cn2, transition_type=transform),
                    ReplacementTransform(tree_latex1, tree_latex2, run_time=0.001),
                ] + second_phase
                self.play(
                    *second_phase,
                    path_arc=0 * DEGREES,
                    rate_func=smooth,
                    run_time=transition.duration * SHAPE_TRANSFORMATION_MULTIPLIER,
                )

                ## Third phase:
                if l2 != l2_post_color or cn2 != cn2_post_color:
                    self.play(
                        transform(l2, l2_post_color),
                        transform(cn2, cn2_post_color),
                        path_arc=0 * DEGREES,
                        rate_func=linear,
                        run_time=transition.duration * COLOR_FADE_OUT_MULTIPLIER,
                    )

                # Clean up phase:
                self.wait(transition.duration * INTERMISSION_MULTIPLIER)
                if is_last or transition.event != transitions[i + 1].event:
                    self.remove(descriptor_latex)

        self.wait(2)

    def create_plot(self, tree, color=WHITE, z_index=0, extra_hash=None):
        k = (
            to_string(
                tree,
                full_precision=True,
                feature_map=self.feature_map,
                options=self.options,
            ),
            color,
            z_index,
            extra_hash,
        )
        if "create_plot" not in self.CACHE:
            self.CACHE["create_plot"] = {}
        if k in self.CACHE["create_plot"]:
            return self.CACHE["create_plot"][k]

        dx = 0.001
        x_range = [-5, 5]
        test_x = np.linspace(*x_range, 1000)
        test_y = get_y(test_x)
        test_z = get_z(test_x)
        test_out = self.true_f(test_x, test_y, test_z)

        y_range = [np.min(test_out), np.max(test_out)]
        diff = y_range[1] - y_range[0]
        pct_expand = 0.5
        y_range = [y_range[0] - pct_expand * diff, y_range[1] + pct_expand * diff]
        assert np.array(y_range).shape == (2,)

        desired_y_range = [-5, 5]
        multiplier = (desired_y_range[1] - desired_y_range[0]) / (
            y_range[1] - y_range[0]
        )
        offset = (y_range[0] + y_range[1]) / 2

        def predict_f(x):
            y = get_y(x)
            z = get_z(x)
            if isinstance(x, np.ndarray):
                cX = np.array([x, y, z]).astype(np.float64)
                assert cX.shape[0] == 3
                assert len(cX.shape) == 2
                out = sr.eval_tree_array(tree, cX, self.options)[0]
                # out = np.clip(out, *y_range)
                return out
            else:
                cX = np.array([[x], [y], [z]]).astype(np.float64)
                assert cX.shape[0] == 3
                assert len(cX.shape) == 2
                out = sr.eval_tree_array(tree, cX, self.options)[0]
                # out = np.clip(out, *y_range)
                return out[0]

        predicted_test_x = np.linspace(*x_range, 1000)
        predicted_test_out = predict_f(predicted_test_x)
        # Find x_range that is in bounds.
        predicted_x_range = [0.0, 0.0]
        for xi, oi in zip(predicted_test_x[500:], predicted_test_out[500:]):
            if oi >= y_range[0] and oi <= y_range[1]:
                predicted_x_range[1] = max(xi, predicted_x_range[0])
            else:
                break
        for xi, oi in zip(predicted_test_x[:500][::-1], predicted_test_out[:500][::-1]):
            if oi >= y_range[0] and oi <= y_range[1]:
                predicted_x_range[0] = min(xi, predicted_x_range[1])
            else:
                break

        axes = Axes(
            x_range=x_range,
            y_range=desired_y_range,
            x_length=10,
            y_length=10,
            tips=False,
            axis_config={"stroke_width": 1, "color": GRAY},
            z_index=z_index + PLOT_Z_INDEX,
        )
        truth_graph = axes.plot(
            lambda x: np.clip(
                multiplier * (self.true_f(x, get_y(x), get_z(x)) - offset),
                desired_y_range[0] - 1,
                desired_y_range[1] + 1,
            ),
            x_range=x_range + [dx],
            color=BLUE,
            stroke_width=2,
            use_vectorized=True,
            z_index=z_index + PLOT_Z_INDEX + 1,
        )
        prediction_graph = axes.plot(
            lambda x: np.clip(
                multiplier * (predict_f(x) - offset),
                desired_y_range[0] - 1,
                desired_y_range[1] + 1,
            ),
            x_range=predicted_x_range + [dx],
            color=color,
            stroke_width=2,
            use_vectorized=True,
            z_index=z_index + PLOT_Z_INDEX + 1,
        )

        legend_labels = [
            (r"\text{Truth}", BLUE),
            (r"\text{Prediction}", color),
        ]
        legend = VGroup()
        spacing = 0.4
        for i, (label_text, label_color) in enumerate(legend_labels):
            line = Line(
                start=[0, 0, 0], end=[0.5, 0, 0], stroke_width=2, color=label_color
            ).set_z_index(z_index + PLOT_Z_INDEX + 2)
            text = (
                Tex(label_text, color=label_color)
                .next_to(line, RIGHT)
                .set_z_index(z_index + PLOT_Z_INDEX + 2)
            )
            legend_item = VGroup(line, text)
            legend.add(legend_item)
            if i > 0:
                legend.arrange(DOWN, buff=spacing)

        plot = VGroup(axes, truth_graph, prediction_graph, legend.shift(LEGEND_OFFSET))
        out = plot.move_to(START_POS + PLOT_OFFSET * GLOBAL_SCALE).scale(
            GLOBAL_SCALE * 0.9
        )

        self.CACHE["create_plot"][k] = out
        return out

    def create_description(self, text, z_index=0, color=DESCRIPTION_COLOR):
        k = (text, z_index, str(color))
        if "create_description" not in self.CACHE:
            self.CACHE["create_description"] = {}
        if k in self.CACHE["create_description"]:
            return self.CACHE["create_description"][k]
        v = (
            MathTex(r"\text{" + text + "}")
            .set_z_index(z_index + TEXT_Z_INDEX)
            .move_to(START_POS + DESCRIPTION_OFFSET * GLOBAL_SCALE)
            .scale(GLOBAL_SCALE * TEXT_SCALE * DESCRIPTION_SCALE)
            .set_color(color)
        )
        self.CACHE["create_description"][k] = v
        return v

    def create_equation(self, tree) -> MathTex:
        k = to_string(
            tree,
            full_precision=True,
            feature_map=self.feature_map,
            options=self.options,
        )
        if "create_equation" not in self.CACHE:
            self.CACHE["create_equation"] = {}
        if k in self.CACHE["create_equation"]:
            return self.CACHE["create_equation"][k]
        v = (
            MathTex(to_string(tree, feature_map=self.feature_map, options=self.options))
            .set_z_index(TEXT_Z_INDEX)
            .move_to(START_POS + EQUATION_OFFSET * GLOBAL_SCALE)
            .scale(GLOBAL_SCALE * TEXT_SCALE * 1.3)
        )
        self.CACHE["create_equation"][k] = v
        return v

    def create_tree(
        self,
        tree_data,
        scale,
        highlight,
        z_index,
        pos=START_POS,
        default_color=None,
        default_line_color=None,
        default_stroke_width=DEFAULT_STROKE,
        extra_hash=None,
    ):
        k = to_string(
            tree_data,
            full_precision=True,
            feature_map=self.feature_map,
            options=self.options,
        )
        k = (
            k,
            str(pos),
            str(scale),
            str(highlight),
            z_index,
            default_color,
            default_line_color,
            default_stroke_width,
            extra_hash,
        )
        if "create_tree" not in self.CACHE:
            self.CACHE["create_tree"] = {}
        if k in self.CACHE["create_tree"]:
            return self.CACHE["create_tree"][k]

        def create_node(tree, pos, scale, depth=0, highlight=None):
            color = (
                RED
                if (highlight and "" in highlight)
                else (default_color if default_color else WHITE)
            )
            if tree.degree == 0:
                circle_node = self.default_circle_node(
                    to_string(tree, feature_map=self.feature_map, options=self.options),
                    pos,
                    scale,
                    color,
                    z_index,
                )
                return VGroup(), VGroup(circle_node)
            else:
                op = str(jl.just_op_to_string(tree, self.options))
                children = [tree.l, tree.r] if tree.degree == 2 else [tree.l]
                circle_node = self.default_circle_node(op, pos, scale, color, z_index)
                branching = jl.count_branching(tree)

                remove_l = re.compile(r"^l\.?")
                remove_r = re.compile(r"^r\.?")

                if tree.degree == 2:
                    child_positions = [
                        pos + scale * (LEFT * (branching - 1) + DOWN * 2),
                        pos + scale * (RIGHT * (branching - 1) + DOWN * 2),
                    ]
                    if highlight:
                        child_highlights = [
                            [
                                remove_l.sub("", h)
                                for h in highlight
                                if h.startswith("l")
                            ],
                            [
                                remove_r.sub("", h)
                                for h in highlight
                                if h.startswith("r")
                            ],
                        ]
                    else:
                        child_highlights = [None, None]
                else:
                    child_positions = [pos + scale * (DOWN * 2)]
                    if highlight:
                        child_highlights = [
                            [
                                remove_l.sub("", h)
                                for h in highlight
                                if h.startswith("l")
                            ]
                        ]
                    else:
                        child_highlights = [None]

                child_nodes = [
                    create_node(child, child_pos, scale, depth + 1, h)
                    for child, child_pos, h in zip(
                        children, child_positions, child_highlights
                    )
                ]

                # Create line but offset by the radius of the circles, so it doesn't overlap:
                def create_line(child_pos):
                    circle_radius = scale * 0.5
                    start_pos = pos + circle_radius * (
                        child_pos - pos
                    ) / np.linalg.norm(child_pos - pos)
                    end_pos = child_pos - circle_radius * (
                        child_pos - pos
                    ) / np.linalg.norm(child_pos - pos)
                    return Line(
                        start_pos,
                        end_pos,
                        stroke_width=default_stroke_width * GLOBAL_SCALE,
                        z_index=z_index + LINE_Z_INDEX,
                    ).set_color(default_line_color if default_line_color else GREY)

                lines = [create_line(child_pos) for child_pos in child_positions]
                lines_group, circle_nodes_group = zip(*child_nodes)
                return (
                    VGroup(*lines, *lines_group),
                    VGroup(circle_node, *circle_nodes_group),
                )

        v = create_node(tree_data, pos, scale, highlight=highlight)
        self.CACHE["create_tree"][k] = v
        return v

    def default_circle(self, scale, color, z_index):
        return Circle(
            radius=scale * 0.5,
            stroke_width=DEFAULT_STROKE * GLOBAL_SCALE,
            fill_opacity=1,
            fill_color=BACKGROUND_COLOR,
            color=color,
        ).set_z_index(z_index + CIRCLE_Z_INDEX)

    def default_circle_node(self, text, pos, scale, color, z_index):
        # TODO: Should I do caching here?
        node = (
            MathTex(text)
            .move_to(pos)
            .scale(scale * TEXT_SCALE)
            .set_z_index(z_index + TEXT_Z_INDEX)
            .set_color(color)
        )
        if re.search(r"\d", text):
            node = node.scale(0.8)

        circle = self.default_circle(scale, color, z_index).move_to(node.get_center())
        return VGroup(node, circle)

    def set_background(self):
        background = Rectangle(
            width=14.2,
            height=8,
            stroke_width=0,
            fill_color=BACKGROUND_COLOR,
            fill_opacity=1,
        )
        self.add(background)


class Event:
    # ["tree", "mutation", "highlight", "tree_gen", "get_const_history", "type"],
    # defaults=(None, None, None, None, None, "mutation"),
    def __init__(
        self,
        tree=None,
        mutation=None,
        highlight=None,
        tree_gen=None,
        get_const_history=None,
        type="mutation",
    ):
        self.tree = tree
        self.mutation = mutation
        self.highlight = highlight
        self.tree_gen = tree_gen
        self.get_const_history = get_const_history
        self.type = type
        # Create random hash:
        self.hash = "".join(
            random.choice(string.ascii_letters + string.digits) for _ in range(10)
        )


def gen_transition_sequence(events):
    transitions = []
    ltree = None
    ctree = None
    for i, event in enumerate(events):
        if i == 0:
            ctree = event.tree
            transitions.append(Expression(ctree))
        elif event.type == "mutation":
            assert event.tree is not None, f"Event: ${event}"
            if isinstance(event.tree, tuple):
                assert ltree is not None, f"Event: ${event}"
                ctree = event.tree[0](ltree)
            else:
                ctree = sr.copy_node(event.tree)
            transitions.append(
                ExpressionTransition(
                    ltree,
                    ctree,
                    mutation=event.mutation,
                    highlight=event.highlight,
                    perturb=event.mutation != "Simplify",
                    event=event.hash,
                )
            )
        elif event.type == "optimization":
            assert event.tree_gen is not None, f"Event: ${event}"
            const_history = event.get_const_history(sr.copy_node(ltree))
            nt = len(const_history)
            for i, c in enumerate(const_history):
                ctree = sr.copy_node(event.tree_gen(c))
                duration = DEFAULT_TRANSITION_TIME / nt
                transitions.append(
                    ExpressionTransition(
                        ltree,
                        ctree,
                        duration=duration,
                        transition_type=partial(ReplacementTransform, run_time=0.0001),
                        mutation="Constant optimization",
                        event=event.hash,
                        highlight=(
                            (event.highlight if i > 0 else None),
                            event.highlight,
                            event.highlight,
                            (event.highlight if i < nt - 1 else None),
                        ),
                        perturb=False,
                    )
                )

                ltree = sr.copy_node(ctree)
        else:
            raise ValueError(f"Unknown event type {type}")

        ltree = sr.copy_node(ctree)

    return transitions
