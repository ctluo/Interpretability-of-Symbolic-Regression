"""Generate all numerical results and figures used by Manuscript_revised.tex.

The script is deterministic.  It uses independent Latin-hypercube designs for
training, validation, and testing; fits three transparent enumerative symbolic
model families and four comparison baselines; evaluates physical constraints;
and quantifies preference sensitivity with a Jansen total-order estimator.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "interpretable_sol_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy.stats import qmc
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "revision_assets"
ASSET_DIR.mkdir(exist_ok=True)

ALPHA_MIN, ALPHA_MAX = -10.0, 15.0
MACH_MIN, MACH_MAX = 0.1, 0.8
TRAIN_SEED, VALID_SEED, TEST_SEED = 42, 43, 44
NOISE_SEED = 420
SIGMA_TRAIN = 0.03


def target(alpha: np.ndarray, mach: np.ndarray) -> np.ndarray:
    """Author-constructed response used only for the illustrative benchmark."""
    alpha_rad = np.pi * alpha / 180.0
    return (
        2.0
        * np.pi
        * alpha_rad
        / np.sqrt(1.0 + 0.2 * mach**2)
        * (1.0 - 0.0015 * alpha**2)
    )


def lhs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    u = qmc.LatinHypercube(d=2, seed=seed).random(n)
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * u[:, 0]
    mach = MACH_MIN + (MACH_MAX - MACH_MIN) * u[:, 1]
    return alpha, mach


@dataclass
class Model:
    name: str
    family: str
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]
    expression: str
    complexity: int


def least_squares_model(
    name: str,
    family: str,
    design: Callable[[np.ndarray, np.ndarray], np.ndarray],
    term_names: list[str],
    alpha: np.ndarray,
    mach: np.ndarray,
    y: np.ndarray,
    complexity: int,
) -> Model:
    beta, *_ = np.linalg.lstsq(design(alpha, mach), y, rcond=None)

    def predict(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        return design(np.asarray(a), np.asarray(m)) @ beta

    expression = " + ".join(
        f"({coef:.8g})*{term}" for coef, term in zip(beta, term_names)
    )
    return Model(name, family, predict, expression, complexity)


def normalized_inputs(alpha: np.ndarray, mach: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            (alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN),
            (mach - MACH_MIN) / (MACH_MAX - MACH_MIN),
        ]
    )


def fit_models(
    a_train: np.ndarray,
    m_train: np.ndarray,
    y_train: np.ndarray,
    a_valid: np.ndarray,
    m_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[list[Model], dict[str, object]]:
    compact = least_squares_model(
        "ESR-compact",
        "enumerative SR",
        lambda a, m: np.column_stack([np.ones_like(a), a]),
        ["1", "alpha"],
        a_train,
        m_train,
        y_train,
        complexity=5,
    )
    polynomial = least_squares_model(
        "ESR-polynomial",
        "enumerative SR",
        lambda a, m: np.column_stack([a, a**3, m**2]),
        ["alpha", "alpha^3", "M^2"],
        a_train,
        m_train,
        y_train,
        complexity=11,
    )

    # The grammar contains alpha/sqrt(1+c M^2) and alpha^3.  The structural
    # constant c is selected on validation data from a declared fixed grid.
    c_grid = np.linspace(0.05, 0.40, 36)
    pa_candidates: list[tuple[float, float, Model]] = []
    for c in c_grid:
        design = lambda a, m, c=c: np.column_stack(
            [a / np.sqrt(1.0 + c * m**2), a**3]
        )
        model = least_squares_model(
            "ESR-physics-aware",
            "physics-aware enumerative SR",
            design,
            [f"alpha/sqrt(1+{c:.4g}*M^2)", "alpha^3"],
            a_train,
            m_train,
            y_train,
            complexity=14,
        )
        rmse = float(np.sqrt(np.mean((model.predict(a_valid, m_valid) - y_valid) ** 2)))
        pa_candidates.append((rmse, c, model))
    pa_rmse, pa_c, physics_aware = min(pa_candidates, key=lambda item: item[0])

    # Gaussian Nadaraya-Watson/RBF smoother.  Bandwidth is selected on validation.
    x_train = normalized_inputs(a_train, m_train)
    bandwidth_grid = np.array([0.05, 0.08, 0.12, 0.18, 0.25])

    def rbf_predict_for_h(a: np.ndarray, m: np.ndarray, h: float) -> np.ndarray:
        x = normalized_inputs(np.asarray(a), np.asarray(m))
        d2 = np.sum((x[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
        w = np.exp(-d2 / (2.0 * h**2))
        return (w @ y_train) / np.maximum(w.sum(axis=1), 1.0e-15)

    rbf_scores = []
    for h in bandwidth_grid:
        pred = rbf_predict_for_h(a_valid, m_valid, float(h))
        rbf_scores.append(float(np.sqrt(np.mean((pred - y_valid) ** 2))))
    best_h = float(bandwidth_grid[int(np.argmin(rbf_scores))])
    rbf = Model(
        "RBF",
        "black-box smoother",
        lambda a, m: rbf_predict_for_h(a, m, best_h),
        f"Gaussian RBF smoother, h={best_h:.3g}",
        complexity=len(a_train),
    )

    # Sparse polynomial library.  Scaling is fitted on training data only.
    def sparse_library(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [a, a**2, a**3, m, m**2, a * m, a * m**2, a**2 * m]
        )

    sparse_terms = ["alpha", "alpha^2", "alpha^3", "M", "M^2", "alpha*M", "alpha*M^2", "alpha^2*M"]
    # Scale magnitudes without centering so that fit_intercept=False genuinely
    # preserves the zero-output boundary when every library term vanishes.
    scaler = StandardScaler(with_mean=False).fit(sparse_library(a_train, m_train))
    x_sparse = scaler.transform(sparse_library(a_train, m_train))
    lasso = LassoCV(
        alphas=np.logspace(-6, -2, 80),
        cv=5,
        fit_intercept=False,
        max_iter=100000,
        random_state=TRAIN_SEED,
    ).fit(x_sparse, y_train)
    coef_original = lasso.coef_ / scaler.scale_
    intercept_from_scaling = 0.0

    def sparse_predict(a: np.ndarray, m: np.ndarray) -> np.ndarray:
        return lasso.predict(scaler.transform(sparse_library(np.asarray(a), np.asarray(m))))

    nonzero = np.flatnonzero(np.abs(coef_original) > 1.0e-10)
    sparse_expr_parts = [
        f"({coef_original[i]:.8g})*{sparse_terms[i]}" for i in nonzero
    ]
    sparse = Model(
        "Sparse",
        "sparse regression",
        sparse_predict,
        " + ".join(sparse_expr_parts),
        complexity=int(max(1, 3 * len(nonzero))),
    )

    # Three-leaf regression tree, trained on the same noisy training data.
    tree = DecisionTreeRegressor(
        max_leaf_nodes=3, min_samples_leaf=20, random_state=TRAIN_SEED
    ).fit(np.column_stack([a_train, m_train]), y_train)
    rule_text = export_text(tree, feature_names=["alpha", "M"], decimals=5)
    rule = Model(
        "Rule/tree",
        "three-leaf regression tree",
        lambda a, m: tree.predict(np.column_stack([np.asarray(a), np.asarray(m)])),
        rule_text.replace("\n", " | "),
        complexity=3,
    )

    # A theory-constrained one-parameter baseline; coefficient fitted on train.
    phi_train = a_train / np.sqrt(1.0 + 0.2 * m_train**2)
    k_phys = float(np.dot(phi_train, y_train) / np.dot(phi_train, phi_train))
    physics = Model(
        "Physics",
        "physics-informed formula",
        lambda a, m: k_phys * np.asarray(a) / np.sqrt(1.0 + 0.2 * np.asarray(m) ** 2),
        f"({k_phys:.8g})*alpha/sqrt(1+0.2*M^2)",
        complexity=10,
    )

    details = {
        "physics_aware_c": pa_c,
        "physics_aware_validation_rmse": pa_rmse,
        "rbf_bandwidth": best_h,
        "lasso_alpha": float(lasso.alpha_),
        "tree_rules": rule_text,
        "physics_coefficient": k_phys,
    }
    return [compact, polynomial, physics_aware, rbf, sparse, rule, physics], details


def evaluate_model(
    model: Model,
    a_test: np.ndarray,
    m_test: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | str | bool]:
    pred = model.predict(a_test, m_test)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    r2 = float(r2_score(y_test, pred))

    mach_grid = np.linspace(MACH_MIN, MACH_MAX, 121)
    boundary = float(np.max(np.abs(model.predict(np.zeros_like(mach_grid), mach_grid))))

    alpha_grid = np.linspace(-5.0, 10.0, 121)
    aa, mm = np.meshgrid(alpha_grid, mach_grid, indexing="ij")
    da = 1.0e-4
    derivative = (
        model.predict((aa + da).ravel(), mm.ravel())
        - model.predict((aa - da).ravel(), mm.ravel())
    ) / (2.0 * da)
    monotonicity = float(np.mean(derivative >= -1.0e-8))

    # Lipschitz estimate in normalized coordinates on a regular joint-domain grid.
    alpha_lip = np.linspace(ALPHA_MIN, ALPHA_MAX, 81)
    mach_lip = np.linspace(MACH_MIN, MACH_MAX, 61)
    ag, mg = np.meshgrid(alpha_lip, mach_lip, indexing="ij")
    ha = 1.0e-4 * (ALPHA_MAX - ALPHA_MIN)
    hm = 1.0e-4 * (MACH_MAX - MACH_MIN)
    grad_a_norm = (
        model.predict((ag + ha).ravel(), mg.ravel())
        - model.predict((ag - ha).ravel(), mg.ravel())
    ) / (2.0e-4)
    grad_m_norm = (
        model.predict(ag.ravel(), (mg + hm).ravel())
        - model.predict(ag.ravel(), (mg - hm).ravel())
    ) / (2.0e-4)
    lipschitz = float(np.max(np.sqrt(grad_a_norm**2 + grad_m_norm**2)))

    # Expected absolute response change under 1% normalized input perturbations.
    idx = rng.choice(len(a_test), size=3000, replace=True)
    a0, m0 = a_test[idx], m_test[idx]
    eps = rng.normal(0.0, 0.01, size=(len(idx), 2))
    a1 = np.clip(a0 + eps[:, 0] * (ALPHA_MAX - ALPHA_MIN), ALPHA_MIN, ALPHA_MAX)
    m1 = np.clip(m0 + eps[:, 1] * (MACH_MAX - MACH_MIN), MACH_MIN, MACH_MAX)
    perturbation = float(np.mean(np.abs(model.predict(a1, m1) - model.predict(a0, m0))))

    feasible = bool(boundary <= 0.02 and monotonicity >= 0.99)
    return {
        "model": model.name,
        "family": model.family,
        "rmse": rmse,
        "r2": r2,
        "complexity": model.complexity,
        "boundary_max": boundary,
        "monotonicity": monotonicity,
        "lipschitz_normalized": lipschitz,
        "perturbation_sensitivity": perturbation,
        "feasible_baseline": feasible,
        "expression": model.expression,
    }


def rmse_bootstrap_ci(
    y: np.ndarray, pred: np.ndarray, seed: int, n_boot: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    residual2 = (pred - y) ** 2
    indices = rng.integers(0, len(y), size=(n_boot, len(y)))
    values = np.sqrt(np.mean(residual2[indices], axis=1))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def pareto_flags(rows: list[dict[str, object]]) -> dict[str, bool]:
    feasible = [row for row in rows if bool(row["feasible_baseline"])]
    flags = {str(row["model"]): False for row in rows}
    for i, row in enumerate(feasible):
        x = np.array(
            [float(row["rmse"]), float(row["complexity"]), float(row["perturbation_sensitivity"])]
        )
        dominated = False
        for j, other in enumerate(feasible):
            if i == j:
                continue
            z = np.array(
                [float(other["rmse"]), float(other["complexity"]), float(other["perturbation_sensitivity"])]
            )
            if np.all(z <= x) and np.any(z < x):
                dominated = True
                break
        flags[str(row["model"])] = not dominated
    return flags


def preference_analysis(rows: list[dict[str, object]], n_power: int = 11) -> dict[str, object]:
    """Jansen total-order indices for an ESR-physics-aware selection margin."""
    names = [str(row["model"]) for row in rows]
    target_index = names.index("ESR-physics-aware")
    rmse = np.array([float(row["rmse"]) for row in rows])
    complexity = np.array([float(row["complexity"]) for row in rows])
    stability = np.array([float(row["perturbation_sensitivity"]) for row in rows])
    boundary = np.array([float(row["boundary_max"]) for row in rows])
    monotonicity = np.array([float(row["monotonicity"]) for row in rows])

    parameter_names = [
        "accuracy preference",
        "complexity preference",
        "stability preference",
        "complexity scale",
        "stability scale",
        "boundary threshold",
        "monotonicity threshold",
    ]

    def transform(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        latent = -np.log(np.clip(unit[:, :3], 1.0e-12, 1.0))
        weights = latent / latent.sum(axis=1, keepdims=True)
        c_scale = 10.0 + 30.0 * unit[:, 3]
        s_scale = 0.01 + 0.05 * unit[:, 4]
        b_threshold = 0.005 + 0.025 * unit[:, 5]
        m_threshold = 0.97 + 0.03 * unit[:, 6]
        return weights, c_scale, s_scale, b_threshold, m_threshold

    def evaluate(unit: np.ndarray, return_winner: bool = False):
        weights, c_scale, s_scale, b_threshold, m_threshold = transform(unit)
        pred_score = 1.0 - np.minimum(rmse[None, :] / 0.25, 1.0)
        comp_score = 1.0 - np.minimum(complexity[None, :] / c_scale[:, None], 1.0)
        stab_score = 1.0 - np.minimum(stability[None, :] / s_scale[:, None], 1.0)
        utility = (
            weights[:, [0]] * pred_score
            + weights[:, [1]] * comp_score
            + weights[:, [2]] * stab_score
        )
        feasible = (boundary[None, :] <= b_threshold[:, None]) & (
            monotonicity[None, :] >= m_threshold[:, None]
        )
        utility = np.where(feasible, utility, -1.0)
        winner = np.argmax(utility, axis=1)
        other = utility.copy()
        other[:, target_index] = -1.0
        margin = utility[:, target_index] - np.max(other, axis=1)
        return (margin, winner) if return_winner else margin

    n = 2**n_power
    a = qmc.Sobol(d=7, scramble=True, seed=731).random_base2(n_power)
    b = qmc.Sobol(d=7, scramble=True, seed=947).random_base2(n_power)
    ya = evaluate(a)
    yb = evaluate(b)
    variance = float(np.var(np.concatenate([ya, yb]), ddof=1))
    total_order = []
    for i in range(7):
        ab = a.copy()
        ab[:, i] = b[:, i]
        yab = evaluate(ab)
        total_order.append(float(np.mean((ya - yab) ** 2) / (2.0 * variance)))

    sample = np.vstack([a, b])
    margin, winner = evaluate(sample, return_winner=True)
    probabilities = {name: float(np.mean(winner == i)) for i, name in enumerate(names)}
    return {
        "parameter_names": parameter_names,
        "total_order": total_order,
        "top1_probability": probabilities,
        "target_margin_median": float(np.median(margin)),
        "target_margin_q025": float(np.quantile(margin, 0.025)),
        "target_margin_q975": float(np.quantile(margin, 0.975)),
        "base_sample_size": n,
        "total_model_evaluations": int(n * (2 + 7)),
    }


def save_table(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.5)


def make_case_figures(
    models: list[Model],
    rows: list[dict[str, object]],
    a_test: np.ndarray,
    m_test: np.ndarray,
    y_test: np.ndarray,
    gsa: dict[str, object],
) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.3), constrained_layout=True)
    for ax, model, color, row in zip(axes.ravel(), models, colors, rows):
        pred = model.predict(a_test, m_test)
        ax.scatter(y_test, pred, s=7, alpha=0.28, color=color, edgecolors="none")
        lo = min(float(y_test.min()), float(pred.min()))
        hi = max(float(y_test.max()), float(pred.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.9, linestyle="--")
        ax.set_title(f"{model.name}\nRMSE={float(row['rmse']):.4f}", fontsize=9)
        ax.set_xlabel("True $C_L$")
        ax.set_ylabel("Predicted $C_L$")
        style_axes(ax)
    axes.ravel()[-1].axis("off")
    fig.savefig(ASSET_DIR / "case_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "case_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    selected = [models[0], models[1], models[2], models[3], models[4], models[6]]
    alpha_line = np.linspace(ALPHA_MIN, ALPHA_MAX, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True, constrained_layout=True)
    for ax, mach_value in zip(axes, [0.1, 0.45, 0.8]):
        m_line = np.full_like(alpha_line, mach_value)
        ax.plot(alpha_line, target(alpha_line, m_line), color="black", linewidth=2.2, label="Target")
        for model, color in zip(selected, colors):
            ax.plot(alpha_line, model.predict(alpha_line, m_line), color=color, linewidth=1.1, label=model.name)
        ax.axhline(0.0, color="0.6", linewidth=0.6)
        ax.axvline(0.0, color="0.6", linewidth=0.6)
        ax.set_title(f"Mach = {mach_value:.2f}")
        ax.set_xlabel(r"Angle of attack $\alpha$ (deg)")
        style_axes(ax)
    axes[0].set_ylabel("$C_L$")
    axes[-1].legend(fontsize=7, ncol=2, loc="best")
    fig.savefig(ASSET_DIR / "case_lift_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "case_lift_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)
    for row, color in zip(rows, colors):
        marker = "o" if bool(row["feasible_baseline"]) else "x"
        ax.scatter(float(row["complexity"]), float(row["rmse"]), s=65, color=color, marker=marker)
        ax.annotate(str(row["model"]), (float(row["complexity"]), float(row["rmse"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Declared structural complexity")
    ax.set_ylabel("Test RMSE (log scale)")
    ax.set_title("Accuracy--complexity map after physical checks")
    style_axes(ax)
    fig.savefig(ASSET_DIR / "case_pareto.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "case_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    probs = gsa["top1_probability"]
    prob_names = list(probs.keys())
    prob_values = [float(probs[name]) for name in prob_names]
    axes[0].barh(prob_names, prob_values, color=colors)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Top-rank probability")
    axes[0].set_title("Preference/threshold uncertainty")
    style_axes(axes[0])
    st = np.asarray(gsa["total_order"], dtype=float)
    pnames = list(gsa["parameter_names"])
    order = np.argsort(st)
    axes[1].barh(np.asarray(pnames)[order], st[order], color="#2878B5")
    axes[1].set_xlabel("Jansen total-order index")
    axes[1].set_title("Sensitivity of physics-aware ESR margin")
    style_axes(axes[1])
    fig.savefig(ASSET_DIR / "rank_uncertainty.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "rank_uncertainty.pdf", bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, face="#EAF2F8", edge="#21618C", fontsize=9):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.3,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.25, color="#34495E"))


def make_conceptual_figures(best_name: str, best_rmse: float, best_r2: float) -> None:
    # Scientific two-layer taxonomy without decorative icons.
    fig, ax = plt.subplots(figsize=(12.0, 5.4), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "Two-layer organization of symbolic-model assessment", ha="center", fontsize=15, weight="bold")
    ax.text(0.03, 0.72, "Layer A\nInterpretability\nassessment", ha="left", va="center", fontsize=11, weight="bold", color="#154360")
    labels = [
        "Structural & cognitive\naccessibility",
        "Variable attribution\n& interaction",
        "Physical & knowledge\nconsistency",
        "Numerical & structural\nrobustness",
        "Human utility\n& auditability",
    ]
    for i, label in enumerate(labels):
        rounded_box(ax, (0.19 + i * 0.155, 0.62), 0.135, 0.20, label, fontsize=8.5)
    ax.plot([0.03, 0.97], [0.52, 0.52], color="#7F8C8D", lw=0.8)
    ax.text(0.03, 0.30, "Layer B\nDecision &\nselection", ha="left", va="center", fontsize=11, weight="bold", color="#641E16")
    decision = ["Predictive\nperformance", "Hard feasibility\nfilters", "Pareto\nanalysis", "Rank uncertainty\n& GSA"]
    for i, label in enumerate(decision):
        rounded_box(ax, (0.22 + i * 0.18, 0.20), 0.145, 0.20, label, face="#FDEDEC", edge="#922B21", fontsize=8.5)
    ax.text(0.5, 0.07, "Interpretability is profiled; deployment choice is a separate, auditable decision.", ha="center", fontsize=9.5, style="italic")
    fig.savefig(ASSET_DIR / "taxonomy.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "taxonomy.pdf", bbox_inches="tight")
    plt.close(fig)

    # Framework flowchart with explicit rejection branch.
    fig, ax = plt.subplots(figsize=(13.0, 5.6), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0.02, 0.65), 0.13, 0.18, "Data, domain\nknowledge &\noperational domain")
    rounded_box(ax, (0.19, 0.65), 0.12, 0.18, "Generate\nsymbolic\ncandidates")
    rounded_box(ax, (0.35, 0.65), 0.12, 0.18, "Simplify &\ncheck expression\ndomain")
    diamond = Polygon([[0.56, 0.88], [0.64, 0.74], [0.56, 0.60], [0.48, 0.74]], closed=True, edgecolor="#922B21", facecolor="#FDEDEC", lw=1.3)
    ax.add_patch(diamond)
    ax.text(0.56, 0.74, "Hard physical\nconstraints?", ha="center", va="center", fontsize=8.5)
    rounded_box(ax, (0.68, 0.65), 0.12, 0.18, "Compute separate\nperformance &\ninterpretability profiles")
    rounded_box(ax, (0.84, 0.65), 0.12, 0.18, "Pareto\nfiltering")
    rounded_box(ax, (0.68, 0.23), 0.12, 0.18, "Preference &\nthreshold GSA")
    rounded_box(ax, (0.84, 0.23), 0.12, 0.18, "Expert review &\nfinal admissible\nmodel set")
    rounded_box(ax, (0.42, 0.23), 0.12, 0.18, "Reject model &\nrecord audit\nreason", face="#F4F6F7", edge="#616A6B")
    arrow(ax, (0.15, 0.74), (0.19, 0.74))
    arrow(ax, (0.31, 0.74), (0.35, 0.74))
    arrow(ax, (0.47, 0.74), (0.48, 0.74))
    arrow(ax, (0.64, 0.74), (0.68, 0.74))
    arrow(ax, (0.80, 0.74), (0.84, 0.74))
    arrow(ax, (0.90, 0.65), (0.75, 0.41))
    arrow(ax, (0.80, 0.32), (0.84, 0.32))
    arrow(ax, (0.56, 0.60), (0.50, 0.41))
    ax.text(0.655, 0.77, "pass", fontsize=8, color="#196F3D")
    ax.text(0.515, 0.52, "fail", fontsize=8, color="#922B21")
    ax.annotate("", xy=(0.25, 0.65), xytext=(0.42, 0.32), arrowprops=dict(arrowstyle="->", lw=1.0, linestyle="--", color="#7F8C8D"))
    ax.text(0.29, 0.45, "revise search/constraints", fontsize=8, color="#7F8C8D", rotation=18)
    fig.savefig(ASSET_DIR / "framework_flowchart.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "framework_flowchart.pdf", bbox_inches="tight")
    plt.close(fig)

    # Separate graphical abstract in a wide, compact format.
    fig, ax = plt.subplots(figsize=(13.5, 4.3), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0.02, 0.22), 0.23, 0.58, "Literature evidence\n+\nSymbolic expressions\nEngineering constraints", face="#E8F6F3", edge="#117864", fontsize=11)
    rounded_box(ax, (0.38, 0.22), 0.25, 0.58, "Interpretability profile\n+\nHard feasibility first\nPareto comparison\nRanking sensitivity", face="#EBF5FB", edge="#1F618D", fontsize=11)
    rounded_box(ax, (0.76, 0.22), 0.22, 0.58, f"Auditable selection\n+\n{best_name}\nRMSE = {best_rmse:.4f}\n$R^2$ = {best_r2:.5f}", face="#FDEDEC", edge="#922B21", fontsize=11)
    arrow(ax, (0.25, 0.51), (0.38, 0.51))
    arrow(ax, (0.63, 0.51), (0.76, 0.51))
    fig.savefig(ASSET_DIR / "graphical_abstract.png", dpi=300, bbox_inches="tight")
    fig.savefig(ASSET_DIR / "graphical_abstract.pdf", bbox_inches="tight")
    plt.close(fig)


def write_latex_macros(rows: list[dict[str, object]], details: dict[str, object], gsa: dict[str, object]) -> None:
    by_name = {str(row["model"]): row for row in rows}
    best = by_name["ESR-physics-aware"]
    top_prob = float(gsa["top1_probability"]["ESR-physics-aware"])
    lines = [
        "% Auto-generated by generate_revision_assets.py; do not edit manually.",
        f"\\newcommand{{\\CaseTrainN}}{{160}}",
        f"\\newcommand{{\\CaseValidN}}{{80}}",
        f"\\newcommand{{\\CaseTestN}}{{1000}}",
        f"\\newcommand{{\\BestModelName}}{{ESR-physics-aware}}",
        f"\\newcommand{{\\BestRMSE}}{{{float(best['rmse']):.4f}}}",
        f"\\newcommand{{\\BestRtwo}}{{{float(best['r2']):.5f}}}",
        f"\\newcommand{{\\BestBoundary}}{{{float(best['boundary_max']):.4f}}}",
        f"\\newcommand{{\\BestMonotonicity}}{{{100*float(best['monotonicity']):.1f}\\%}}",
        f"\\newcommand{{\\BestTopProbability}}{{{100*top_prob:.1f}\\%}}",
        f"\\newcommand{{\\GsaBaseN}}{{{int(gsa['base_sample_size'])}}}",
        f"\\newcommand{{\\GsaEvalN}}{{{int(gsa['total_model_evaluations'])}}}",
        f"\\newcommand{{\\SelectedC}}{{{float(details['physics_aware_c']):.3f}}}",
        f"\\newcommand{{\\SelectedBandwidth}}{{{float(details['rbf_bandwidth']):.3f}}}",
    ]
    (ASSET_DIR / "revision_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    a_train, m_train = lhs(160, TRAIN_SEED)
    a_valid, m_valid = lhs(80, VALID_SEED)
    a_test, m_test = lhs(1000, TEST_SEED)
    rng_noise = np.random.default_rng(NOISE_SEED)
    y_train_clean = target(a_train, m_train)
    y_train = y_train_clean + rng_noise.normal(0.0, SIGMA_TRAIN, len(a_train))
    y_valid = target(a_valid, m_valid)
    y_test = target(a_test, m_test)

    models, details = fit_models(a_train, m_train, y_train, a_valid, m_valid, y_valid)
    metric_rng = np.random.default_rng(990)
    rows = [evaluate_model(model, a_test, m_test, y_test, metric_rng) for model in models]
    for i, (row, model) in enumerate(zip(rows, models)):
        low, high = rmse_bootstrap_ci(y_test, model.predict(a_test, m_test), seed=2000 + i)
        row["rmse_ci_low"] = low
        row["rmse_ci_high"] = high
    flags = pareto_flags(rows)
    for row in rows:
        row["pareto_feasible"] = flags[str(row["model"])]

    gsa = preference_analysis(rows)
    save_table(rows, ASSET_DIR / "case_metrics.csv")
    np.savetxt(
        ASSET_DIR / "case_data.csv",
        np.column_stack([a_test, m_test, y_test]),
        delimiter=",",
        header="alpha_deg,Mach,C_L_true",
        comments="",
    )
    (ASSET_DIR / "model_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ASSET_DIR / "gsa_results.json").write_text(
        json.dumps(gsa, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_latex_macros(rows, details, gsa)
    make_case_figures(models, rows, a_test, m_test, y_test, gsa)

    feasible_rows = [row for row in rows if bool(row["feasible_baseline"])]
    best = min(feasible_rows, key=lambda row: float(row["rmse"]))
    make_conceptual_figures(str(best["model"]), float(best["rmse"]), float(best["r2"]))

    print("Generated revision assets in", ASSET_DIR)
    for row in rows:
        print(
            f"{row['model']:18s} RMSE={float(row['rmse']):.6f} "
            f"R2={float(row['r2']):.6f} boundary={float(row['boundary_max']):.5f} "
            f"mono={float(row['monotonicity']):.3f} feasible={row['feasible_baseline']} "
            f"pareto={row['pareto_feasible']}"
        )
    print("Top-rank probabilities:", gsa["top1_probability"])


if __name__ == "__main__":
    main()
