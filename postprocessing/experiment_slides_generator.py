#!/usr/bin/env python3
"""
General experiment slide generator for step, gaussian, and tanh controllers.

Examples
--------
most used one:

/usr/bin/python3 ../experiment_slides_generator.py \
  --csv results/04-09-26/alpha10-2_T1_v2/merged_alpha10-2_T1_v2.csv \
  --video data/04-09-26/alpha10-2_T1_v2.mp4 \
  --controller tanh \
  -o results/04-09-26/alpha10-2_T1_v2/alpha10-2_T1_v2


  ---

/usr/bin/python3 experiment_slides_generator.py \
  --csv results/18-06-26/merged.csv \
  --video data/18-06-26/experiment.mp4 \
  --controller tanh \
  -o results/experiment_tanh

/usr/bin/python3 experiment_slides_generator.py \
  --csv results/18-06-26/merged.csv \
  --video data/18-06-26/experiment.mp4 \
  --controller gaussian \
  -o results/experiment_gaussian

/usr/bin/python3 experiment_slides_generator.py \
  --csv results/18-06-26/merged.csv \
  --video data/18-06-26/experiment.mp4 \
  --controller step \
  -o results/experiment_step
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib import rc
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.ticker import MaxNLocator

LIGHT = 20
DARK = 80

AZADEH_Y_EDGES = np.array([0.0, 36.75, 73.5, 110.25, 147.0])
AZADEH_REGION_LABELS = (
    "Light far from boundary",
    "Light close to boundary",
    "Dark close to boundary",
    "Dark far from boundary",
)
# Ordered as: light far, light close, dark close, dark far.
AZADEH_REGION_COLORS = ("#8F2D2D", "#E69F00", "#7FB3D5", "#2F6DB0")
# Binary Azadeh regions are ordered from dark/far to light/far so the legend
# follows the physical progression across the tilted light/dark boundary.
AZADEH_BINARY_REGION_LABELS = (
    "Dark far from boundary",
    "Dark close to boundary",
    "Light close to boundary",
    "Light far from boundary",
)
AZADEH_BINARY_REGION_COLORS = (
    AZADEH_REGION_COLORS[3],
    AZADEH_REGION_COLORS[2],
    AZADEH_REGION_COLORS[1],
    AZADEH_REGION_COLORS[0],
)
AZADEH_BINARY_POLICY_EDGES = np.array([-0.5, 50.0, 100.5])
AZADEH_BINARY_ARENA_CENTER_CM = np.array([75.0, 75.0])
AZADEH_BINARY_ARENA_RADIUS_CM = 75.0
AZADEH_BINARY_CLOSE_DISTANCE_CM = 15.0
AZADEH_BINARY_ROBOT_RADIUS_CM = 2.75

# Fitted to the light/dark transition in data/21-10-26/bkg.bmp. Tracking
# frames are vertically flipped before x/y are converted to centimetres, so
# this is the boundary in the same coordinate system as the merged CSV:
# y = 70.833 + 0.044774 * x (cm). Positive signed distance is the dark side.
AZADEH_BINARY_BOUNDARY_SLOPE = 0.044774
AZADEH_BINARY_BOUNDARY_INTERCEPT_CM = 70.833
AZADEH_BINARY_BOUNDARY_POINT_CM = np.array(
    [0.0, AZADEH_BINARY_BOUNDARY_INTERCEPT_CM]
)
AZADEH_BINARY_BOUNDARY_NORMAL = np.array(
    [-AZADEH_BINARY_BOUNDARY_SLOPE, 1.0]
)
AZADEH_BINARY_BOUNDARY_NORMAL /= np.linalg.norm(AZADEH_BINARY_BOUNDARY_NORMAL)
AZADEH_D_M = 0.1
AZADEH_TAU_D_MEAN_SEC = 30.0
AZADEH_TAU_R_MAX_SEC = 10.0
AZADEH_BINARY_TAU_R_MAX_SEC = 100.0
AZADEH_MUTATION_COEFFICIENT = AZADEH_D_M * AZADEH_TAU_D_MEAN_SEC
AZADEH_BINARY_PROFILE_TRACE_TIMES_MIN = (5.0, 25.0)
AZADEH_BINARY_PROFILE_SMOOTHING_WINDOW = 5
AZADEH_BINARY_PROFILE_Y_BINS = 60

CONTROLLER_DEFAULTS = {
    "step": {
        "policy_run_time_plot": 100.0,
        "tau_r_max_plot": 100.0,
    },
    "gaussian": {
        "tau_r_max_plot": 100.0,
        "tau_r_0_plot": 100.0,
        "sigma_policy": 10.0,
    },
    "tanh": {
        "tau_r_max_plot": 100.0,
        "alpha": 1.0,
        "delta": 7.5,
    },
}


def configure_style(no_tex: bool = False) -> None:
    use_tex = (shutil.which("latex") is not None) and (not no_tex)
    rc("text", usetex=use_tex)
    rc("font", family="serif")
    rc("axes", labelsize=12)
    rc("axes", titlesize=13)
    rc("legend", fontsize=9)
    rc("xtick", labelsize=10)
    rc("ytick", labelsize=10)
    rc("figure", dpi=120)
    rc("savefig", dpi=300)
    plt.rcParams.update({
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "mathtext.fontset": "cm",
    })


def make_muted_cmap(name: str, low_color: str, high_color: str):
    return LinearSegmentedColormap.from_list(name, [low_color, high_color], N=256)


def theme_colors():
    return {
        "density_line": "#4C78A8",
        "density_cmap": make_muted_cmap("density_soft_blue", "#E8EEF5", "#4C78A8"),
        "packing_cmap": LinearSegmentedColormap.from_list(
            "packing_soft_bwr", ["#2166AC", "#FFFFFF", "#D73027"], N=256
        ),
        "weight_line": "#5B8E55",
        "weight_cmap": make_muted_cmap("weight_soft_green", "#EDF4EA", "#5B8E55"),
        "score_line": "#C06C5B",
        "score_cmap": make_muted_cmap("score_soft_red", "#F7ECE8", "#C06C5B"),
        "weight_traces_cmap": "Greens",
        "score_traces_cmap": "Reds",
        "controller": "#4F5B66",
        "phase": "#6B7280",
        "robust_light_band": "#F3F4F6",
        "robust_curve_colors": ["#2E7D32", "#66A95C", "#A5D6A7"],
    }


def parse_edges_arg(text):
    try:
        edges = np.array([float(v.strip()) for v in text.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Edges must be a comma-separated list of floats.") from exc
    if edges.size < 2:
        raise argparse.ArgumentTypeError("At least two edges are required.")
    if not np.all(np.isfinite(edges)):
        raise argparse.ArgumentTypeError("All edges must be finite.")
    if np.any(np.diff(edges) <= 0):
        raise argparse.ArgumentTypeError("Edges must be strictly increasing.")
    return edges


def load_tracking_data(csv_path: Path) -> pd.DataFrame:
    usecols = ["t", "x", "y", "theta", "q", "w", "id"]
    dtype = {
        "t": "float64",
        "x": "float64",
        "y": "float64",
        "theta": "float64",
        "q": "float64",
        "w": "float64",
        "id": "int64",
    }
    df = pd.read_csv(csv_path, usecols=usecols, dtype=dtype)
    missing = set(usecols).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["t", "x", "y", "id"]).copy()
    df = df.drop_duplicates(subset=["t", "id"]).copy()
    if df.empty:
        raise ValueError("No valid rows remain after cleaning the CSV.")
    return df.sort_values(["t", "id"]).reset_index(drop=True)


def read_fraction_light_from_df(df: pd.DataFrame, light_threshold: float, light_region: str = "below") -> pd.DataFrame:
    work = df[["t", "y", "id"]].copy()
    if light_region == "below":
        work["in_light"] = work["y"] <= light_threshold
    elif light_region == "above":
        work["in_light"] = work["y"] >= light_threshold
    else:
        raise ValueError("light_region must be 'below' or 'above'")
    summary = (
        work.groupby("t", sort=True)
        .agg(n_light=("in_light", "sum"), n_tot=("id", "size"))
        .reset_index()
    )
    summary["fraction_light"] = summary["n_light"] / summary["n_tot"]
    return summary


def get_initial_weights(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.sort_values(["id", "t"])
        .groupby("id", sort=True)
        .first()
        .reset_index()[["id", "w", "t"]]
        .rename(columns={"t": "t0"})
    )
    out["w"] = out["w"].clip(0, 100)
    return out


def assign_azadeh_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign observations to the four fixed Azadeh y-axis regions."""
    work = df[["t", "y", "w", "q"]].copy()
    work = work.dropna(subset=["t", "y"])
    y = work["y"].to_numpy(dtype=float)
    valid = (y >= AZADEH_Y_EDGES[0]) & (y <= AZADEH_Y_EDGES[-1])
    work = work.loc[valid].copy()
    work["region_idx"] = np.digitize(
        work["y"].to_numpy(dtype=float), AZADEH_Y_EDGES[1:-1], right=False
    )
    work["region"] = work["region_idx"].map(dict(enumerate(AZADEH_REGION_LABELS)))
    return work


def build_azadeh_region_fraction(df: pd.DataFrame) -> pd.DataFrame:
    work = assign_azadeh_regions(df)
    times = np.sort(df["t"].dropna().unique())
    totals = df.groupby("t", sort=True).size().reindex(times, fill_value=0)
    counts = work.groupby(["t", "region"], sort=True).size().unstack("region", fill_value=0)
    counts = counts.reindex(index=times, columns=AZADEH_REGION_LABELS, fill_value=0)
    result = counts.div(totals.replace(0, np.nan), axis=0).fillna(0.0)
    result.index.name = "t"
    return result.reset_index()


def build_azadeh_region_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = assign_azadeh_regions(df)
    times = np.sort(df["t"].dropna().unique())
    means = work.groupby(["t", "region"], sort=True)[value_col].mean().unstack("region")
    means = means.reindex(index=times, columns=AZADEH_REGION_LABELS)
    means.index.name = "t"
    return means.reset_index()


def assign_azadeh_binary_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Assign observations to four zones defined by the tilted boundary."""
    work = df[["t", "x", "y", "w", "q"]].copy()
    work = work.dropna(subset=["t", "x", "y"])
    x = work["x"].to_numpy(dtype=float)
    y = work["y"].to_numpy(dtype=float)
    dx = x - AZADEH_BINARY_ARENA_CENTER_CM[0]
    dy = y - AZADEH_BINARY_ARENA_CENTER_CM[1]
    inside = dx * dx + dy * dy <= AZADEH_BINARY_ARENA_RADIUS_CM**2
    work = work.loc[inside].copy()
    signed_distance = (
        np.column_stack((x[inside], y[inside]))
        - AZADEH_BINARY_BOUNDARY_POINT_CM
    ) @ AZADEH_BINARY_BOUNDARY_NORMAL
    work["boundary_distance_cm"] = signed_distance
    work["region"] = np.select(
        [
            signed_distance > AZADEH_BINARY_CLOSE_DISTANCE_CM,
            signed_distance > 0.0,
            signed_distance >= -AZADEH_BINARY_CLOSE_DISTANCE_CM,
        ],
        AZADEH_BINARY_REGION_LABELS[:3],
        default=AZADEH_BINARY_REGION_LABELS[3],
    )
    return work


def _circle_area_on_positive_side(signed_line_offset: float, radius: float) -> float:
    """Area of a circle on the side ``u >= signed_line_offset``."""
    if signed_line_offset <= -radius:
        return float(np.pi * radius**2)
    if signed_line_offset >= radius:
        return 0.0
    root = np.sqrt(radius**2 - signed_line_offset**2)
    return float(radius**2 * np.arccos(signed_line_offset / radius) - signed_line_offset * root)


def build_azadeh_binary_zone_areas() -> dict[str, float]:
    """Return the exact area of each four-zone circle/line intersection."""
    radius = AZADEH_BINARY_ARENA_RADIUS_CM
    center_offset = float(
        np.dot(
            AZADEH_BINARY_ARENA_CENTER_CM - AZADEH_BINARY_BOUNDARY_POINT_CM,
            AZADEH_BINARY_BOUNDARY_NORMAL,
        )
    )

    def area_with_distance_at_least(threshold: float) -> float:
        return _circle_area_on_positive_side(threshold - center_offset, radius)

    area_dark_far = area_with_distance_at_least(AZADEH_BINARY_CLOSE_DISTANCE_CM)
    area_dark = area_with_distance_at_least(0.0)
    area_dark_close = area_dark - area_dark_far
    area_light_close = area_with_distance_at_least(-AZADEH_BINARY_CLOSE_DISTANCE_CM) - area_dark
    area_total = float(np.pi * radius**2)
    area_light_far = area_total - area_with_distance_at_least(-AZADEH_BINARY_CLOSE_DISTANCE_CM)
    return {
        AZADEH_BINARY_REGION_LABELS[0]: area_dark_far,
        AZADEH_BINARY_REGION_LABELS[1]: area_dark_close,
        AZADEH_BINARY_REGION_LABELS[2]: area_light_close,
        AZADEH_BINARY_REGION_LABELS[3]: area_light_far,
    }


def build_azadeh_binary_packing_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """Build zone packing fractions using the actual circle/line zone areas."""
    work = assign_azadeh_binary_regions(df)
    times = np.sort(df["t"].dropna().unique())
    counts = work.groupby(["t", "region"], sort=True).size().unstack("region", fill_value=0)
    counts = counts.reindex(index=times, columns=AZADEH_BINARY_REGION_LABELS, fill_value=0)
    area_by_region = build_azadeh_binary_zone_areas()
    robot_area = np.pi * AZADEH_BINARY_ROBOT_RADIUS_CM**2
    result = counts.astype(float).copy()
    for label in AZADEH_BINARY_REGION_LABELS:
        result[label] *= robot_area / area_by_region[label]
    result.index.name = "t"
    return result.reset_index()


def build_azadeh_binary_region_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = assign_azadeh_binary_regions(df)
    times = np.sort(df["t"].dropna().unique())
    means = work.groupby(["t", "region"], sort=True)[value_col].mean().unstack("region")
    means = means.reindex(index=times, columns=AZADEH_BINARY_REGION_LABELS)
    means.index.name = "t"
    return means.reset_index()


def build_azadeh_binary_y_profiles(
    df: pd.DataFrame,
    t_edges: np.ndarray,
    y_edges: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build time-resolved y profiles for density/packing fraction and score.

    The density profiles are averaged over the x direction by dividing the
    mean number of robots in each horizontal strip by that strip's arena
    area.  Fast and slow profiles use the total population as their common
    reference, so their sum is the total density for binary policies.
    """
    if metric not in {"density", "packing_fraction"}:
        raise ValueError("metric must be 'density' or 'packing_fraction'")

    work = df[["t", "x", "y", "q", "w", "id"]].copy()
    work = work.dropna(subset=["t", "x", "y"])
    inside = (
        (work["x"] - AZADEH_BINARY_ARENA_CENTER_CM[0]) ** 2
        + (work["y"] - AZADEH_BINARY_ARENA_CENTER_CM[1]) ** 2
        <= AZADEH_BINARY_ARENA_RADIUS_CM**2
    )
    work = work.loc[inside].copy()
    work["t_idx"] = pd.cut(
        work["t"], bins=t_edges, labels=False, include_lowest=True, right=True
    )
    work["y_idx"] = pd.cut(
        work["y"], bins=y_edges, labels=False, include_lowest=True, right=True
    )
    work = work.dropna(subset=["t_idx", "y_idx"]).copy()
    if work.empty:
        raise RuntimeError("No samples fall inside the selected profile ranges.")

    work["t_idx"] = work["t_idx"].astype(np.int32)
    work["y_idx"] = work["y_idx"].astype(np.int16)
    n_t = len(t_edges) - 1
    n_y = len(y_edges) - 1
    full_index = pd.MultiIndex.from_product(
        [np.arange(n_t, dtype=np.int32), np.arange(n_y, dtype=np.int16)],
        names=["t_idx", "y_idx"],
    )

    n_frames = (
        work.groupby("t_idx", sort=False)["t"]
        .nunique()
        .reindex(np.arange(n_t), fill_value=0)
        .to_numpy(dtype=float)
    )
    strip_areas = _circular_strip_areas(
        y_edges,
        AZADEH_BINARY_ARENA_CENTER_CM[1],
        AZADEH_BINARY_ARENA_RADIUS_CM,
    )
    robot_area = np.pi * AZADEH_BINARY_ROBOT_RADIUS_CM**2
    profiles = np.full((n_t, n_y, 3), np.nan, dtype=float)
    score_profiles = np.full((n_t, n_y, 3), np.nan, dtype=float)

    group_masks = (
        np.ones(len(work), dtype=bool),
        np.isclose(work["w"].to_numpy(dtype=float), 100.0),
        np.isclose(work["w"].to_numpy(dtype=float), 0.0),
    )
    for group_index, mask in enumerate(group_masks):
        group = work.loc[mask]
        counts = (
            group.groupby(["t_idx", "y_idx"], sort=False)
            .size()
            .reindex(full_index, fill_value=0)
            .to_numpy(dtype=float)
            .reshape(n_t, n_y)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_counts = counts / n_frames[:, None]
            if metric == "density":
                profiles[:, :, group_index] = mean_counts / strip_areas[None, :]
            else:
                profiles[:, :, group_index] = (
                    mean_counts * robot_area / strip_areas[None, :]
                )

        # Match the score-profile analysis used in
        # results/15-10-26/score_azadeh_profile/analysis.py: first compute a
        # mean score for each robot in each y-bin, then average those robot
        # means. The time-resolved binary slide applies that same weighting
        # independently inside each 30-second time bin, so robots with fewer
        # observations do not receive less weight merely because of tracking
        # gaps.
        per_robot_scores = (
            group.dropna(subset=["q"])
            .groupby(["t_idx", "y_idx", "id"], sort=False)["q"]
            .mean()
            .groupby(["t_idx", "y_idx"], sort=False)
            .mean()
        )
        score_profiles[:, :, group_index] = (
            per_robot_scores
            .reindex(full_index)
            .to_numpy(dtype=float)
            .reshape(n_t, n_y)
        )

    profiles[n_frames <= 0, :, :] = np.nan
    profiles[:, strip_areas <= 0.0, :] = np.nan
    return profiles, score_profiles


def step_running_time(I, w, policy_run_time_plot):
    I = np.asarray(I, dtype=float)
    tau = np.where(I <= float(w), float(policy_run_time_plot), 0.0)
    return np.clip(tau, 0.0, float(policy_run_time_plot))


def gaussian_running_time(I, w, tau_r_max, tau_r_0, sigma_policy):
    I = np.asarray(I, dtype=float)
    sigma = float(sigma_policy)
    if sigma <= 0:
        raise ValueError("sigma_policy must be strictly positive.")
    tau = float(tau_r_max) - float(tau_r_0) * np.exp(-((I - float(w)) ** 2) / (sigma ** 2))
    return np.clip(tau, 0.0, float(tau_r_max))


def tanh_running_time(I, w, tau_r_max, alpha, delta):
    I = np.asarray(I, dtype=float)
    tau = float(tau_r_max) * (
        1.0 + 0.5 * (
            np.tanh(float(alpha) * (I - (float(w) + float(delta))))
            - np.tanh(float(alpha) * (I - (float(w) - float(delta))))
        )
    )
    return np.clip(tau, 0.0, float(tau_r_max))


def build_controller_params(args):
    defaults = CONTROLLER_DEFAULTS[args.controller].copy()
    params = defaults.copy()
    if args.policy_run_time_plot is not None:
        params["policy_run_time_plot"] = float(args.policy_run_time_plot)
    if args.tau_r_max_plot is not None:
        params["tau_r_max_plot"] = float(args.tau_r_max_plot)
    if args.tau_r_0_plot is not None:
        params["tau_r_0_plot"] = float(args.tau_r_0_plot)
    if args.sigma_policy is not None:
        params["sigma_policy"] = float(args.sigma_policy)
    if args.alpha is not None:
        params["alpha"] = float(args.alpha)
    if args.delta is not None:
        params["delta"] = float(args.delta)
    return params


def controller_running_time(I, w, controller: str, params: dict):
    if controller == "step":
        return step_running_time(I, w, params["policy_run_time_plot"])
    if controller == "gaussian":
        return gaussian_running_time(I, w, params["tau_r_max_plot"], params["tau_r_0_plot"], params["sigma_policy"])
    if controller == "tanh":
        return tanh_running_time(I, w, params["tau_r_max_plot"], params["alpha"], params["delta"])
    raise ValueError(f"Unsupported controller: {controller}")


def is_good_and_robust_policy(w: float, controller: str, params: dict, robust_light_min: int, robust_light_max: int, robust_max_runtime: float) -> bool:
    I_values = np.arange(int(robust_light_min), int(robust_light_max) + 1, dtype=float)
    tau = controller_running_time(I_values, w, controller, params)
    return bool(np.all(tau < float(robust_max_runtime)))


def centers_to_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 0:
        raise ValueError("Cannot build edges from an empty centers array.")
    if centers.size == 1:
        dt = 0.5
        return np.array([centers[0] - dt, centers[0] + dt], dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate(([first], mids, [last]))


def build_time_edges(df, t_bins=None, t_min=None, t_max=None, t_bin_width=20.0, t_edges=None):
    t_valid = df["t"].to_numpy()
    if t_edges is not None:
        edges = np.asarray(t_edges, dtype=float)
    else:
        if t_min is None:
            t_min = float(np.nanmin(t_valid))
        if t_max is None:
            t_max = float(np.nanmax(t_valid))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            raise RuntimeError("Invalid t range.")
        if t_bin_width is not None:
            if t_bin_width <= 0:
                raise ValueError("--t-bin-width must be > 0.")
            n_t = max(1, int(np.ceil((t_max - t_min) / t_bin_width)))
            edges = t_min + np.arange(n_t + 1, dtype=float) * t_bin_width
            if edges[-1] < t_max:
                edges = np.append(edges, t_max)
            else:
                edges[-1] = t_max
        elif t_bins is not None:
            if t_bins <= 0:
                raise ValueError("--t-bins must be > 0.")
            edges = np.linspace(t_min, t_max, int(t_bins) + 1)
        else:
            times = np.sort(df["t"].unique())
            edges = centers_to_edges(times)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise RuntimeError("Invalid time-bin edges.")
    return edges


def build_relative_density_map(rho_map: np.ndarray, y_bins: int) -> tuple[np.ndarray, float]:
    rho_0 = 1.0 / float(y_bins)
    rel_map = (rho_map / rho_0) - 1.0
    return rel_map, rho_0


def _circular_strip_areas(y_edges: np.ndarray, center_y: float, radius: float) -> np.ndarray:
    """Return the area of each horizontal strip inside a circular arena."""
    lower = np.clip(y_edges[:-1], center_y - radius, center_y + radius)
    upper = np.clip(y_edges[1:], center_y - radius, center_y + radius)
    lower_u = lower - center_y
    upper_u = upper - center_y

    def primitive(u: np.ndarray) -> np.ndarray:
        root = np.sqrt(np.maximum(radius**2 - u**2, 0.0))
        return u * root + radius**2 * np.arcsin(np.clip(u / radius, -1.0, 1.0))

    return np.maximum(primitive(upper_u) - primitive(lower_u), 0.0)


def build_azadeh_binary_local_packing_fraction_map(
    mean_count_map: np.ndarray, y_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert mean robot counts per y-bin into local packing fraction."""
    strip_areas = _circular_strip_areas(
        y_edges,
        AZADEH_BINARY_ARENA_CENTER_CM[1],
        AZADEH_BINARY_ARENA_RADIUS_CM,
    )
    robot_area = np.pi * AZADEH_BINARY_ROBOT_RADIUS_CM**2
    with np.errstate(divide="ignore", invalid="ignore"):
        packing_map = mean_count_map * robot_area / strip_areas[None, :]
    packing_map[:, strip_areas <= 0.0] = np.nan
    return packing_map, strip_areas


def build_maps_fast(df, y_bins=20, y_min=None, y_max=None, t_bins=None, t_min=None, t_max=None, t_bin_width=20.0, t_edges=None):
    y_valid = df["y"].to_numpy()
    if y_min is None:
        y_min = float(np.nanmin(y_valid))
    if y_max is None:
        y_max = float(np.nanmax(y_valid))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        raise RuntimeError("Invalid y range.")

    y_edges = np.linspace(y_min, y_max, int(y_bins) + 1)
    t_edges = build_time_edges(df, t_bins=t_bins, t_min=t_min, t_max=t_max, t_bin_width=t_bin_width, t_edges=t_edges)
    n_t = len(t_edges) - 1
    n_b = len(y_edges) - 1

    work = df[["t", "y", "q", "w"]].copy()
    work["t_idx"] = pd.cut(work["t"], bins=t_edges, labels=False, include_lowest=True, right=True)
    work["y_bin"] = pd.cut(work["y"], bins=y_edges, labels=False, include_lowest=True, right=True)
    work = work.dropna(subset=["t_idx", "y_bin"]).copy()
    if work.empty:
        raise RuntimeError("No samples fall inside the selected t/y bin ranges.")
    work["t_idx"] = work["t_idx"].astype(np.int32)
    work["y_bin"] = work["y_bin"].astype(np.int16)

    counts = work.groupby(["t_idx", "y_bin"], sort=False).size().rename("count").reset_index()
    totals = work.groupby("t_idx", sort=False).size().rename("n_tot").reset_index()
    counts = counts.merge(totals, on="t_idx", how="left")
    counts["rho"] = counts["count"] / counts["n_tot"]
    frame_counts = work.groupby("t_idx", sort=False)["t"].nunique().rename("n_frames")
    counts["n_frames"] = counts["t_idx"].map(frame_counts)
    counts["mean_count"] = counts["count"] / counts["n_frames"]

    full_index = pd.MultiIndex.from_product(
        [np.arange(n_t, dtype=np.int32), np.arange(n_b, dtype=np.int16)],
        names=["t_idx", "y_bin"],
    )

    rho_map = (
        counts.set_index(["t_idx", "y_bin"])["rho"]
        .reindex(full_index, fill_value=0.0)
        .to_numpy()
        .reshape(n_t, n_b)
    )
    mean_count_map = (
        counts.set_index(["t_idx", "y_bin"])["mean_count"]
        .reindex(full_index, fill_value=0.0)
        .to_numpy()
        .reshape(n_t, n_b)
    )
    w_mean = (
        work.dropna(subset=["w"]).groupby(["t_idx", "y_bin"], sort=False)["w"]
        .mean().reindex(full_index).to_numpy().reshape(n_t, n_b)
    )
    q_mean = (
        work.dropna(subset=["q"]).groupby(["t_idx", "y_bin"], sort=False)["q"]
        .mean().reindex(full_index).to_numpy().reshape(n_t, n_b)
    )
    return t_edges, y_edges, rho_map, mean_count_map, w_mean, q_mean


def choose_vrange(data, robust=False):
    vals = data[np.isfinite(data)]
    if vals.size == 0:
        return 0.0, 1.0
    if robust:
        vmin = float(np.nanpercentile(vals, 1))
        vmax = float(np.nanpercentile(vals, 99))
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else 0.05 * abs(vmin)
        return vmin - pad, vmax + pad
    return vmin, vmax


def extract_frame(video_path: Path, time_s: float):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open the video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 20
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
        frame_idx = int(round(float(time_s) * float(fps)))
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("OpenCV failed to read the requested frame.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        used_time = frame_idx / fps
        return frame, used_time
    except Exception:
        try:
            import imageio.v3 as iio
            try:
                meta = iio.immeta(str(video_path), plugin="pyav")
            except Exception:
                meta = iio.immeta(str(video_path))
            fps = meta.get("fps", 30.0)
            try:
                fps = float(fps)
            except Exception:
                fps = 30.0
            if fps <= 0:
                fps = 30.0
            nframes = meta.get("nframes", None)
            frame_idx = int(round(float(time_s) * float(fps)))
            if isinstance(nframes, (int, np.integer)) and nframes > 0:
                frame_idx = max(0, min(frame_idx, int(nframes) - 1))
            else:
                frame_idx = max(0, frame_idx)
            try:
                frame = iio.imread(str(video_path), index=frame_idx, plugin="pyav")
            except Exception:
                frame = iio.imread(str(video_path), index=frame_idx)
            if frame.ndim == 2:
                frame = np.repeat(frame[..., None], 3, axis=2)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            used_time = frame_idx / fps
            return frame, used_time
        except Exception as exc:
            raise RuntimeError(
                "Could not extract a frame from the video. Install either OpenCV (cv2) or imageio with video support."
            ) from exc


def add_panel_tag(ax, tag: str) -> None:
    ax.text(0.01, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")


def add_comm_line_time_series(ax, comm_time_s: float, color: str) -> None:
    ax.axvline(comm_time_s / 60.0, color=color, linestyle="--", linewidth=1.0, alpha=0.8, zorder=3)


def add_comm_line_map(ax, comm_time_s: float, color: str) -> None:
    ax.axhline(comm_time_s / 60.0, color=color, linestyle="--", linewidth=1.0, alpha=0.85, zorder=3)


def build_controller_title(controller: str) -> str:
    return rf"Initial {controller} controllers $w_i(t_0)$"


def parse_experiment_metadata(experiment_id: str | None) -> dict[str, object]:
    """Extract title metadata from experiment IDs.

    Both the original ``alpha10-3`` notation and the explicit
    ``alpha_2x10-3`` notation are accepted.  The latter is useful when the
    coefficient is not one, for example ``alpha_2x10-3`` means
    ``alpha_c = 0.002``.
    """
    if not experiment_id:
        return {}

    stem = Path(str(experiment_id)).stem
    match = re.search(
        r"alpha(?:_(?P<coefficient>\d+(?:\.\d+)?)x10-"
        r"(?P<explicit_power>\d+)|(?P<legacy_base>\d+)-"
        r"(?P<legacy_power>\d+))_T(?P<tq>\d+)"
        r"(?:_D(?P<mutation>\d+))?",
        stem,
    )
    if match is None:
        return {}

    if match.group("coefficient") is not None:
        alpha_c = float(match.group("coefficient")) * 10 ** (
            -int(match.group("explicit_power"))
        )
    else:
        # Experiment IDs use the convention ``alpha10-3`` == 10**(-3).
        # The ``10`` before the hyphen is part of that convention, not a
        # multiplicative coefficient.
        alpha_c = 10 ** (-int(match.group("legacy_power")))

    version_match = re.search(r"(?:^|_)v(?P<version>\d+)(?:_|$)", stem)
    return {
        "t_q": int(match.group("tq")),
        "alpha_c": alpha_c,
        "mutation_coefficient": (
            int(match.group("mutation"))
            if match.group("mutation") is not None
            else AZADEH_MUTATION_COEFFICIENT
        ),
        "version": version_match.group("version") if version_match else None,
    }


def build_default_suptitle(
    controller: str,
    params: dict,
    robust_max_runtime: float,
    experiment_id: str | None = None,
) -> str:
    metadata = parse_experiment_metadata(experiment_id)
    t_q = metadata.get("t_q", robust_max_runtime)
    alpha_c = metadata.get("alpha_c")
    version = metadata.get("version")
    version_suffix = f", v{version}" if version is not None else ""
    alpha_c_text = f"{alpha_c:g}" if alpha_c is not None else None

    if controller == "step":
        alpha_text = alpha_c_text or "0.001"
        return rf"Phototaxis step controller: $T_q = {t_q:g}$, $ \alpha_c = {alpha_text} $, $D_M = 0$, $\tau_R = {params['policy_run_time_plot']:g}$, exchange: 'pure genetic drift'{version_suffix}"
    if controller == "gaussian":
        alpha_text = alpha_c_text or "0.01"
        return (
            rf"Phototaxis gaussian controller: $T_q = {t_q:g}$, "
            rf"$ \alpha_c = {alpha_text}, $"
            rf"$ D_M = 0,$ "
            rf"$\sigma = {params['sigma_policy']:g}$, "
            rf"$\tau_{{R,0}} = {params['tau_r_0_plot']:g}$, "
            rf" exchange: 'pure teaching'{version_suffix}"
        )
    alpha_text = alpha_c_text or "0.0001"
    return (
        rf"Phototaxis tanh controller: $T_q = {t_q:g}$, "
        rf"$\delta = {params['delta']:g}$, $\alpha_C = {alpha_text}$, "
        rf"$\tau_{{R,max}} = {params['tau_r_max_plot']:g}${version_suffix}"
    )


def build_azadeh_suptitle(experiment_id: str | None) -> str:
    metadata = parse_experiment_metadata(experiment_id)
    t_q = metadata.get("t_q", 100)
    alpha_c = metadata.get("alpha_c", 0.001)
    mutation_coefficient = metadata.get(
        "mutation_coefficient", AZADEH_MUTATION_COEFFICIENT
    )
    return (
        r"Controller agnostic to light: "
        rf"$T_q={t_q:g}\,\mathrm{{s}}$, $\alpha_c={alpha_c:g}$, "
        rf"$D_M={mutation_coefficient:g}$, "
        rf"$\tau_{{R,\max}}={AZADEH_TAU_R_MAX_SEC:g}\,\mathrm{{s}}$"
    )


def build_azadeh_binary_suptitle(experiment_id: str | None) -> str:
    metadata = parse_experiment_metadata(experiment_id)
    t_q = metadata.get("t_q", 100)
    alpha_c = metadata.get("alpha_c", 0.001)
    mutation_coefficient = metadata.get(
        "mutation_coefficient", AZADEH_MUTATION_COEFFICIENT
    )
    return (
        r"Controller agnostic to light: "
        rf"$T_q={t_q:g}\,\mathrm{{s}}$, $\alpha_c={alpha_c:g}$, "
        rf"$\tau_M={mutation_coefficient:g}$, "
        rf"$\tau_{{R,\max}}={AZADEH_BINARY_TAU_R_MAX_SEC:g}\,\mathrm{{s}}$, "
        r"binary policies $w\in\{0,100\}$"
    )


def plot_controller_panel(ax, initial_weights: pd.DataFrame, colors: dict, controller: str, params: dict, robust_light_min: int, robust_light_max: int, robust_max_runtime: float) -> None:
    ax.axvspan(robust_light_min, robust_light_max, color=colors["robust_light_band"], alpha=1.0, zorder=0)
    ax.axvline(LIGHT, color="0.75", linestyle="--", linewidth=0.5, zorder=1)
    ax.axvline(DARK, color="0.75", linestyle="--", linewidth=0.5, zorder=1)

    wvals = np.sort(initial_weights["w"].dropna().to_numpy(dtype=float))
    if wvals.size == 0:
        ax.text(0.5, 0.5, "No valid initial weights", transform=ax.transAxes, ha="center", va="center")
        return

    I_grid = np.linspace(0.0, 100.0, 800)
    robust_ws = []
    non_robust_points = []
    for w in wvals:
        tau_at_w = controller_running_time(np.array([w]), w, controller, params)[0]
        if is_good_and_robust_policy(w, controller, params, robust_light_min, robust_light_max, robust_max_runtime):
            robust_ws.append(w)
        else:
            non_robust_points.append((w, tau_at_w))

    if non_robust_points:
        pts = np.asarray(non_robust_points, dtype=float)
        ax.scatter(
            pts[:, 0], pts[:, 1],
            facecolors="none",
            edgecolors=colors["weight_line"],
            linewidths=0.8,
            s=14,
            alpha=0.8,
            zorder=2.0,
        )

    robust_ws = np.array(sorted(robust_ws), dtype=float)
    curve_colors = colors["robust_curve_colors"]
    if robust_ws.size == 1:
        curve_colors = [curve_colors[0]]
    elif robust_ws.size == 2:
        curve_colors = [curve_colors[0], "#81C784"]

    for i, w in enumerate(robust_ws):
        tau = controller_running_time(I_grid, w, controller, params)
        tau_at_w = controller_running_time(np.array([w]), w, controller, params)[0]
        c = curve_colors[min(i, len(curve_colors) - 1)]
        line, = ax.plot(I_grid, tau, color=c, linewidth=2.6, alpha=0.98, zorder=3.5, rasterized=True)
        line.set_path_effects([pe.Stroke(linewidth=4.2, foreground="white", alpha=0.9), pe.Normal()])
        ax.scatter([w], [tau_at_w], color=c, s=30, alpha=1.0, zorder=4.0)

    if robust_ws.size > 0:
        idxs = np.arange(robust_ws.size)
        if robust_ws.size > 6:
            idxs = np.unique(np.round(np.linspace(0, robust_ws.size - 1, 6)).astype(int))
        for j, idx in enumerate(idxs):
            w = robust_ws[idx]
            tau_at_w = controller_running_time(np.array([w]), w, controller, params)[0]
            c = curve_colors[min(idx, len(curve_colors) - 1)]
            #ax.annotate(
                #rf"$w^*={int(round(w))}$",
                #xy=(w, tau_at_w),
                #xytext=(w, min(params.get('tau_r_max_plot', params.get('policy_run_time_plot', 100.0)) - 4, 8 + 6 * (j % 2))),
                #textcoords="data",
                #ha="center",
                #va="bottom",
                #fontsize=9,
                #color=c,
                #arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.9, shrinkA=0, shrinkB=4),
                #zorder=4.2,
            #)

    y_max = float(params.get("tau_r_max_plot", params.get("policy_run_time_plot", 100.0)))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, y_max + 2)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel(r"$I(\vec{r})$")
    ax.set_ylabel(r"$\tau_R$ (s)")
    ax.set_title(build_controller_title(controller), pad=8)
    ax.grid(True, axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_azadeh_initial_weights(ax, initial_weights: pd.DataFrame) -> None:
    """Plot light-agnostic constant policies as horizontal weight lines."""
    work = initial_weights.dropna(subset=["id", "w"]).sort_values("id")
    if work.empty:
        ax.text(0.5, 0.5, "No valid initial weights", transform=ax.transAxes, ha="center", va="center")
        return

    weights = np.rint(work["w"].to_numpy(dtype=float).clip(0.0, 100.0))
    intensity = np.array([0.0, 100.0])
    for weight in weights:
        ax.plot(
            intensity,
            [weight, weight],
            color="#4C78A8",
            linewidth=0.9,
            alpha=0.38,
            zorder=2,
        )
    ax.set_xlim(0.0, 100.0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-3, 103)
    ax.set_xlabel(r"$I(\vec{r})$")
    ax.set_ylabel(r"Initial weight $w_i(t_0)$")
    ax.set_title(r"Initial weights: $\tau_R = w/10$ s", pad=8)
    ax.grid(True, axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    tau_ax = ax.twinx()
    tau_ax.set_ylim(-0.3, 10.3)
    tau_ax.set_ylabel(r"Mean run time $\langle\tau_R\rangle$ (s)")
    tau_ax.spines["top"].set_visible(False)
    tau_ax.grid(False)


def plot_azadeh_binary_initial_weights(
    ax,
    initial_weights: pd.DataFrame,
    policy_colors,
) -> None:
    """Plot the initial population fraction in the two binary policies."""
    work = initial_weights.dropna(subset=["id", "w"])
    if work.empty:
        ax.text(0.5, 0.5, "No valid initial weights", transform=ax.transAxes, ha="center", va="center")
        return

    weights = np.where(work["w"].to_numpy(dtype=float) >= 50.0, 100.0, 0.0)
    centers = np.array([0.0, 100.0])
    fractions = np.array([(weights == value).mean() for value in centers])
    counts = np.array([(weights == value).sum() for value in centers])
    bars = ax.bar(
        centers,
        fractions,
        width=50.0,
        color=policy_colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
        zorder=2,
    )
    for bar, fraction, count in zip(bars, fractions, counts):
        fraction_text = f"{fraction:.0%} (n={count})"
        if plt.rcParams["text.usetex"]:
            fraction_text = fraction_text.replace("%", r"\%")
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            fraction + 0.025,
            fraction_text,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlim(-25.0, 125.0)
    ax.set_xticks(centers, [r"$w=0$", r"$w=100$"])
    ax.set_ylim(0.0, max(1.0, float(fractions.max()) + 0.18))
    ax.set_xlabel(r"Policy weight $w$")
    ax.set_ylabel("Fraction of robots")
    ax.set_title("Initial policy distribution: slow and fast robots", pad=8)
    ax.grid(True, axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_azadeh_region_panel(
    ax,
    series: pd.DataFrame,
    title: str,
    ylabel: str,
    comm_time: float,
    t_min: float,
    t_max: float,
    phase_color: str,
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = False,
    scatter_df: pd.DataFrame | None = None,
    scatter_value_col: str | None = None,
) -> None:
    if scatter_df is not None and scatter_value_col is not None:
        scatter_work = assign_azadeh_regions(scatter_df)
        scatter_work = scatter_work.dropna(subset=["t", scatter_value_col])
        for label, color in zip(AZADEH_REGION_LABELS, AZADEH_REGION_COLORS):
            region_data = scatter_work.loc[scatter_work["region"] == label]
            ax.scatter(
                region_data["t"] / 60.0,
                region_data[scatter_value_col],
                color=color,
                s=1.0,
                alpha=0.025,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )

    for label, color in zip(AZADEH_REGION_LABELS, AZADEH_REGION_COLORS):
        ax.plot(
            series["t"] / 60.0,
            series[label],
            color=color,
            linewidth=1.9,
            label=label,
            zorder=3,
        )
    add_comm_line_time_series(ax, comm_time, phase_color)
    ax.set_xlim(t_min / 60.0, t_max / 60.0)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_title(title)
    ax.set_xlabel("$t$ (min)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(loc="best", fontsize=7, frameon=False, handlelength=1.5)


def plot_azadeh_binary_region_panel(
    ax,
    series: pd.DataFrame,
    title: str,
    ylabel: str,
    comm_time: float,
    t_min: float,
    t_max: float,
    phase_color: str,
    y_limits: tuple[float, float] | None = None,
    show_legend: bool = False,
    scatter_df: pd.DataFrame | None = None,
    scatter_value_col: str | None = None,
    scatter_alpha: float = 0.025,
) -> None:
    if scatter_df is not None and scatter_value_col is not None:
        scatter_work = assign_azadeh_binary_regions(scatter_df)
        scatter_work = scatter_work.dropna(subset=["t", scatter_value_col])
        for label, color in zip(AZADEH_BINARY_REGION_LABELS, AZADEH_BINARY_REGION_COLORS):
            region_data = scatter_work.loc[scatter_work["region"] == label]
            ax.scatter(
                region_data["t"] / 60.0,
                region_data[scatter_value_col],
                color=color,
                s=1.0,
                alpha=scatter_alpha,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )

    for label, color in zip(AZADEH_BINARY_REGION_LABELS, AZADEH_BINARY_REGION_COLORS):
        ax.plot(
            series["t"] / 60.0,
            series[label],
            color=color,
            linewidth=2.0,
            label=label,
            zorder=3,
        )
    add_comm_line_time_series(ax, comm_time, phase_color)
    ax.set_xlim(t_min / 60.0, t_max / 60.0)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_title(title)
    ax.set_xlabel("$t$ (min)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(
            loc="upper right",
            ncol=1,
            fontsize=6.5,
            frameon=False,
            handlelength=1.2,
            labelspacing=0.25,
        )


def plot_snapshot_panel(ax, frame, title: str) -> None:
    ax.imshow(frame, rasterized=True)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_fraction_panel(ax, frac: pd.DataFrame, comm_time: float, t_min: float, t_max: float, color: str, phase_color: str) -> None:
    ax.plot(frac["t"] / 60.0, frac["fraction_light"], color=color, linewidth=2.4, zorder=2)
    add_comm_line_time_series(ax, comm_time, phase_color)
    ax.set_xlim(t_min / 60.0, t_max / 60.0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(r"Fraction in light")
    ax.set_xlabel("$t$ (min)")
    ax.set_ylabel(r"$N_{\mathrm{light}}/N_{\mathrm{tot}}$")
    ax.grid(True, axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_population_traces(ax, df: pd.DataFrame, value_col: str, value_label: str, title: str, mean_color: str, traces_cmap: str, comm_time: float, t_min: float, t_max: float, phase_color: str):
    work = df[["t", "id", value_col]].dropna().copy()
    ids = np.sort(work["id"].unique())
    cmap = plt.get_cmap(traces_cmap)
    colors = cmap(np.linspace(0.35, 0.9, max(len(ids), 2)))
    for color, rid in zip(colors, ids):
        grp = work.loc[work["id"] == rid, ["t", value_col]].sort_values("t")
        ax.plot(grp["t"] / 60.0, grp[value_col], color=color, linewidth=0.7, alpha=0.07, rasterized=True, zorder=1)
    mean_curve = work.groupby("t", sort=True)[value_col].mean().reset_index()
    ax.plot(mean_curve["t"] / 60.0, mean_curve[value_col], color=mean_color, linewidth=2.2, zorder=3)
    add_comm_line_time_series(ax, comm_time, phase_color)
    ax.set_xlim(t_min / 60.0, t_max / 60.0)
    ax.set_title(title)
    ax.set_xlabel("$t$ (min)")
    ax.set_ylabel(value_label)
    ax.grid(True, axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_spacetime_panel(fig, ax, t_edges, y_edges, data, title: str, cmap_name, cbar_label: str, comm_time: float, phase_color: str, robust: bool = False, force_zero_min: bool = False, show_ylabel: bool = False, center_zero: bool = False, center_value: float | None = None):
    vmin, vmax = choose_vrange(data, robust=robust)
    if force_zero_min:
        vmin = 0.0
    if center_zero:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
    cmap = cmap_name.copy() if hasattr(cmap_name, "copy") else plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#F3F3F3")
    norm = None
    if center_value is not None:
        center_value = float(np.clip(center_value, vmin, vmax))
        if vmin < center_value < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
    im = ax.pcolormesh(
        y_edges,
        t_edges / 60.0,
        np.ma.masked_invalid(data),
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        rasterized=True,
    )
    add_comm_line_map(ax, comm_time, phase_color)
    ax.set_title(title, pad=6)
    ax.set_xlabel(r"$y$")
    if show_ylabel:
        ax.set_ylabel("$t$ (min)")
    else:
        ax.tick_params(labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.018, fraction=0.055)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=9)


def plot_binary_weight_spacetime_panel(
    fig,
    ax,
    t_edges,
    y_edges,
    data,
    policy_colors,
    comm_time: float,
    phase_color: str,
    show_ylabel: bool = False,
) -> None:
    """Plot the weight field with exactly one color for each binary policy."""
    binary_data = np.where(
        np.isfinite(data),
        np.where(data >= 50.0, 100.0, 0.0),
        np.nan,
    )
    cmap = ListedColormap(policy_colors)
    norm = BoundaryNorm(AZADEH_BINARY_POLICY_EDGES, cmap.N)
    im = ax.pcolormesh(
        y_edges,
        t_edges / 60.0,
        np.ma.masked_invalid(binary_data),
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    add_comm_line_map(ax, comm_time, phase_color)
    ax.set_title("Binary weight field", pad=6)
    ax.set_xlabel(r"$y$")
    if show_ylabel:
        ax.set_ylabel("$t$ (min)")
    else:
        ax.tick_params(labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.018, fraction=0.055, ticks=[0, 100])
    cbar.ax.set_yticklabels([r"$w=0$", r"$w=100$"])
    cbar.ax.tick_params(labelsize=9)


def _nanmean_profiles(profiles: np.ndarray) -> np.ndarray:
    """Average profile rows without warning when a y-bin has no observations."""
    valid = np.isfinite(profiles)
    counts = valid.sum(axis=0)
    sums = np.nansum(profiles, axis=0)
    return np.divide(
        sums,
        counts,
        out=np.full(profiles.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )


def _moving_average_profile(profile: np.ndarray, window: int) -> np.ndarray:
    """Apply a centered, NaN-aware moving average along the y profile."""
    values = np.asarray(profile, dtype=float)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("The profile smoothing window must be odd.")

    valid = np.isfinite(values)
    kernel = np.ones(int(window), dtype=float)
    sums = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    counts = np.convolve(valid.astype(float), kernel, mode="same")
    return np.divide(
        sums,
        counts,
        out=np.full(values.shape, np.nan, dtype=float),
        where=counts > 0,
    )


def _plot_azadeh_binary_profile_points(
    ax,
    profiles: np.ndarray,
    y_centers: np.ndarray,
    time_centers_min: np.ndarray,
    comm_time_min: float,
    time_norm: Normalize,
    time_cmap,
) -> None:
    """Plot faint, time-colored scatter points for every profile bin."""
    for time_index, time_minute in enumerate(time_centers_min):
        profile = profiles[time_index]
        finite = np.isfinite(profile)
        ax.scatter(
            y_centers[finite],
            profile[finite],
            color=time_cmap(time_norm(float(time_minute))),
            s=16,
            alpha=0.18,
            edgecolors="none",
            zorder=2,
        )


def _select_azadeh_binary_profile_traces(
    time_centers_min: np.ndarray,
) -> list[tuple[float, int]]:
    """Select the closest available time bin for each requested checkpoint."""
    time_min = float(np.nanmin(time_centers_min))
    time_max = float(np.nanmax(time_centers_min))
    if time_centers_min.size > 1:
        half_step = 0.5 * float(np.nanmedian(np.diff(time_centers_min)))
    else:
        half_step = 0.5
    selected: list[tuple[float, int]] = []
    used_indices: set[int] = set()
    for requested_time_min in AZADEH_BINARY_PROFILE_TRACE_TIMES_MIN:
        if (
            requested_time_min < time_min - half_step
            or requested_time_min > time_max + half_step
        ):
            continue
        time_index = int(
            np.argmin(np.abs(time_centers_min - requested_time_min))
        )
        if time_index not in used_indices:
            selected.append((requested_time_min, time_index))
            used_indices.add(time_index)
    return selected


def plot_azadeh_binary_profile_page(
    profiles: np.ndarray,
    score_profiles: np.ndarray,
    t_edges: np.ndarray,
    y_edges: np.ndarray,
    comm_time: float,
    metric: str,
    experiment_id: str,
    time_cmap_name: str = "turbo",
):
    """Create the second PDF page with time-colored y profiles."""
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    time_centers_min = (0.5 * (t_edges[:-1] + t_edges[1:])) / 60.0
    comm_time_min = float(comm_time) / 60.0
    time_cmap = plt.get_cmap(time_cmap_name)
    time_min = float(np.nanmin(time_centers_min))
    time_max = float(np.nanmax(time_centers_min))
    if np.isclose(time_min, time_max):
        time_max = time_min + 1.0
    time_norm = Normalize(
        vmin=time_min,
        vmax=time_max,
    )

    density_profiles = (
        profiles[:, :, 0],
        profiles[:, :, 1],
        profiles[:, :, 2],
        profiles[:, :, 2] - profiles[:, :, 1],
    )
    q_profiles = (
        score_profiles[:, :, 0],
        score_profiles[:, :, 1],
        score_profiles[:, :, 2],
        score_profiles[:, :, 2] - score_profiles[:, :, 1],
    )
    if metric == "density":
        density_ylabel = r"$\rho$ (robots / cm$^2$)"
        density_title = "Density"
    else:
        density_ylabel = r"$\phi$"
        density_title = "Packing fraction"

    specs = (
        (density_profiles, density_title, density_ylabel),
        (q_profiles, "Average score", r"Average score $\langle q \rangle$"),
    )
    titles = (
        "All robots",
        r"Fast robots ($w=100$)",
        r"Slow robots ($w=0$)",
        "Slow − fast",
    )

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 10.2), sharex=True)
    for row, (row_profiles, row_title, row_ylabel) in enumerate(specs):
        comparable_values = np.concatenate(
            [profile[np.isfinite(profile)] for profile in row_profiles[:3]]
        )
        y_min = float(np.nanmin(comparable_values))
        y_max = float(np.nanmax(comparable_values))
        if np.isclose(y_min, y_max):
            padding = 0.05 if np.isclose(y_min, 0.0) else 0.05 * abs(y_min)
        else:
            padding = 0.05 * (y_max - y_min)
        common_ylim = (y_min - padding, y_max + padding)
        for column, (ax, profile, title) in enumerate(
            zip(axes[row], row_profiles, titles)
        ):
            _plot_azadeh_binary_profile_points(
                ax,
                profile,
                y_centers,
                time_centers_min,
                comm_time_min,
                time_norm,
                time_cmap,
            )
            if column < 3:
                ax.set_ylim(*common_ylim)
            ax.set_title(f"{row_title}: {title}")
            ax.set_xlim(float(y_edges[0]), float(y_edges[-1]))
            ax.set_ylabel(row_ylabel if column == 0 else "")
            ax.grid(True, axis="y", color="0.92", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 1:
                ax.set_xlabel(r"$y$ (cm)")

    sm = plt.cm.ScalarMappable(norm=time_norm, cmap=time_cmap)
    sm.set_array([])
    colorbar_axis = fig.add_axes([0.93, 0.18, 0.014, 0.64])
    colorbar = fig.colorbar(sm, cax=colorbar_axis)
    colorbar.set_label("Profile time (min)")
    fig.suptitle(
        f"{density_title} and score profiles",
        y=0.965,
        fontsize=15,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.90,
        top=0.89,
        bottom=0.09,
        wspace=0.30,
        hspace=0.38,
    )
    return fig


def build_parser():
    p = argparse.ArgumentParser(description="Create a clean multi-panel slide figure for step, gaussian, or tanh phototaxis experiments.")
    p.add_argument("--csv", required=True, type=Path, help="CSV file with columns t,x,y,theta,q,w,id")
    p.add_argument("--video", required=True, type=Path, help="MP4 video of the experiment")
    p.add_argument("--controller", choices=["step", "gaussian", "tanh"], default="tanh", help="Controller used in the experiment")
    p.add_argument("-o", "--output", type=Path, default=Path("experiment_slides"), help="Output base path, e.g. results/experiment_slides")

    p.add_argument("--light-threshold", type=float, default=73.5)
    p.add_argument("--light-region", choices=["below", "above"], default="below")
    p.add_argument("--comm-time", type=float, default=300.0)
    p.add_argument("--t-init-shot", type=float, default=300.0)
    p.add_argument("--t-mid-shot", type=float, default=900.0)
    p.add_argument("--t-final-shot", type=float, default=1499.0)

    p.add_argument("--y-bins", type=int, default=20)
    p.add_argument("--y-min", type=float, default=None)
    p.add_argument("--y-max", type=float, default=None)

    tg = p.add_mutually_exclusive_group()
    tg.add_argument("--t-bins", type=int, default=None)
    tg.add_argument("--t-bin-width", type=float, default=20.0)
    tg.add_argument("--t-edges", type=parse_edges_arg, default=None)

    p.add_argument("--t-min", type=float, default=None)
    p.add_argument("--t-max", type=float, default=None)

    p.add_argument("--robust-light-min", type=int, default=67)
    p.add_argument("--robust-light-max", type=int, default=77)
    p.add_argument("--robust-max-runtime", type=float, default=10.0, help="Maximum running time considered robust in panel (a)")

    p.add_argument("--policy-run-time-plot", type=float, default=None, help="Optional override for step controller runtime in seconds")
    p.add_argument("--tau-r-max-plot", type=float, default=None, help="Optional override for tau_R,max in seconds")
    p.add_argument("--tau-r-0-plot", type=float, default=None, help="Optional override for tau_R,0 in seconds")
    p.add_argument("--sigma-policy", type=float, default=None, help="Optional override for gaussian sigma")
    p.add_argument("--alpha", type=float, default=None, help="Optional override for tanh alpha")
    p.add_argument("--delta", type=float, default=None, help="Optional override for tanh delta")

    p.add_argument("--suptitle", type=str, default=None, help="Optional custom figure title")
    azadeh_layout = p.add_mutually_exclusive_group()
    azadeh_layout.add_argument(
        "--azadeh",
        action="store_true",
        help="Use the Azadeh four-region summary layout and parameter title",
    )
    azadeh_layout.add_argument(
        "--azadeh_binary",
        action="store_true",
        help="Use the Azadeh binary-policy summary layout",
    )
    p.add_argument(
        "--azadeh-binary-profile-metric",
        "--azadeh_binary_profile_metric",
        choices=["density", "packing_fraction"],
        default="density",
        help="Metric for the binary y-profile page: areal density or packing fraction",
    )
    p.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment identifier used to derive T_q, alpha_c, and version in the title",
    )
    p.add_argument("--no-tex", action="store_true")
    p.add_argument("--show", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    configure_style(no_tex=args.no_tex)
    colors = theme_colors()
    binary_policy_colors = (
        colors["weight_cmap"](0.35),
        colors["weight_cmap"](1.0),
    )
    params = build_controller_params(args)
    azadeh_layout = args.azadeh or args.azadeh_binary

    df = load_tracking_data(args.csv)
    if args.azadeh:
        region_fraction = build_azadeh_region_fraction(df)
        region_weight = build_azadeh_region_mean(df, "w")
        region_score = build_azadeh_region_mean(df, "q")
    elif args.azadeh_binary:
        region_fraction = build_azadeh_binary_packing_fraction(df)
        region_weight = build_azadeh_binary_region_mean(df, "w")
        region_score = build_azadeh_binary_region_mean(df, "q")
    else:
        frac = read_fraction_light_from_df(df, args.light_threshold, args.light_region)
    initial_weights = get_initial_weights(df)

    t_min_data = float(df["t"].min())
    t_max_data = float(df["t"].max())
    t_min = t_min_data if args.t_min is None else float(args.t_min)
    t_max = t_max_data if args.t_max is None else float(args.t_max)
    if azadeh_layout and args.y_min is None:
        y_min = float(AZADEH_Y_EDGES[0])
    else:
        y_min = float(df["y"].min()) if args.y_min is None else float(args.y_min)
    if azadeh_layout and args.y_max is None:
        y_max = float(AZADEH_Y_EDGES[-1])
    else:
        y_max = float(df["y"].max()) if args.y_max is None else float(args.y_max)

    t_edges, y_edges, rho_map, mean_count_map, w_map, q_map = build_maps_fast(
        df,
        y_bins=args.y_bins,
        y_min=y_min,
        y_max=y_max,
        t_bins=args.t_bins,
        t_min=t_min,
        t_max=t_max,
        # The binary layout uses 30-second temporal bins by default while
        # keeping the CLI override available.
        t_bin_width=(
            30.0
            if args.azadeh_binary and args.t_bin_width == 20.0
            else args.t_bin_width
        ),
        t_edges=args.t_edges,
    )

    if args.azadeh_binary:
        density_map, _ = build_azadeh_binary_local_packing_fraction_map(
            mean_count_map,
            y_edges,
        )
        profile_y_edges = np.linspace(
            y_min,
            y_max,
            AZADEH_BINARY_PROFILE_Y_BINS + 1,
        )
        binary_profiles, binary_score_profiles = build_azadeh_binary_y_profiles(
            df,
            t_edges,
            profile_y_edges,
            args.azadeh_binary_profile_metric,
        )
    else:
        density_map, rho_0 = build_relative_density_map(rho_map, args.y_bins)

    init_frame, init_frame_time = extract_frame(args.video, args.t_init_shot)
    mid_frame, mid_frame_time = extract_frame(args.video, args.t_mid_shot)
    final_frame, final_frame_time = extract_frame(args.video, args.t_final_shot)

    output_base = args.output.with_suffix("") if args.output.suffix else args.output
    output_base.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18.0, 10.2))
    gs = fig.add_gridspec(3, 12, height_ratios=[1.0, 0.92, 1.08], wspace=0.55, hspace=0.48)

    ax_ctrl = fig.add_subplot(gs[0, 0:4])
    shots_gs = gs[0, 4:12].subgridspec(1, 3, wspace=0.04)
    ax_shot0 = fig.add_subplot(shots_gs[0, 0])
    ax_shot1 = fig.add_subplot(shots_gs[0, 1])
    ax_shot2 = fig.add_subplot(shots_gs[0, 2])
    ax_frac = fig.add_subplot(gs[1, 0:4])
    ax_w = fig.add_subplot(gs[1, 4:8])
    ax_q = fig.add_subplot(gs[1, 8:12])
    ax_rho = fig.add_subplot(gs[2, 0:4])
    ax_wmap = fig.add_subplot(gs[2, 4:8])
    ax_qmap = fig.add_subplot(gs[2, 8:12])

    if args.azadeh_binary:
        plot_azadeh_binary_initial_weights(
            ax_ctrl,
            initial_weights,
            binary_policy_colors,
        )
    elif args.azadeh:
        plot_azadeh_initial_weights(ax_ctrl, initial_weights)
    else:
        plot_controller_panel(
            ax_ctrl,
            initial_weights,
            colors,
            controller=args.controller,
            params=params,
            robust_light_min=args.robust_light_min,
            robust_light_max=args.robust_light_max,
            robust_max_runtime=args.robust_max_runtime,
        )

    plot_snapshot_panel(ax_shot0, init_frame, rf"$t = {init_frame_time/60.0:.1f}\,\mathrm{{min}}$")
    plot_snapshot_panel(ax_shot1, mid_frame, rf"$t = {mid_frame_time/60.0:.1f}\,\mathrm{{min}}$")
    plot_snapshot_panel(ax_shot2, final_frame, rf"$t = {final_frame_time/60.0:.1f}\,\mathrm{{min}}$")

    if args.azadeh_binary:
        plot_azadeh_binary_region_panel(
            ax_frac,
            region_fraction,
            "Packing fraction by zone",
            r"$\phi$",
            args.comm_time,
            t_min,
            t_max,
            colors["phase"],
            show_legend=False,
        )
        plot_azadeh_binary_region_panel(
            ax_w,
            region_weight,
            "Average weight by zone",
            r"$\langle w \rangle$",
            args.comm_time,
            t_min,
            t_max,
            colors["phase"],
            y_limits=(0.0, 100.0),
            scatter_df=df,
            scatter_value_col="w",
            show_legend=True,
        )
        plot_azadeh_binary_region_panel(
            ax_q,
            region_score,
            "Average score by zone",
            r"$\langle q \rangle$",
            args.comm_time,
            t_min,
            t_max,
            colors["phase"],
            scatter_df=df,
            scatter_value_col="q",
            scatter_alpha=0.008,
            show_legend=False,
        )
    elif args.azadeh:
        plot_azadeh_region_panel(
            ax_frac,
            region_fraction,
            "Fraction of robots in four regions",
            r"Fraction of robots",
            args.comm_time,
            t_min,
            t_max,
            colors["phase"],
            y_limits=(-0.02, 1.02),
            show_legend=True,
        )
        plot_azadeh_region_panel(
            ax_w,
            region_weight,
            "Average weight by region",
            r"$\langle w \rangle$",
            args.comm_time,
            t_min,
            t_max,
            colors["phase"],
            y_limits=(0.0, 100.0),
            scatter_df=df,
            scatter_value_col="w",
        )
        plot_azadeh_region_panel(
            ax_q,
            region_score,
            "Average score by region",
            r"$\langle q \rangle$",
            args.comm_time,
            t_min,
            t_max,
            colors["phase"],
            scatter_df=df,
            scatter_value_col="q",
        )
    else:
        plot_fraction_panel(ax_frac, frac, args.comm_time, t_min, t_max, colors["density_line"], colors["phase"])
        plot_population_traces(ax_w, df, "w", r"$w$", r"Weight trajectories and mean", colors["weight_line"], colors["weight_traces_cmap"], args.comm_time, t_min, t_max, colors["phase"])
        plot_population_traces(ax_q, df, "q", r"$q$", r"Score trajectories and mean", colors["score_line"], colors["score_traces_cmap"], args.comm_time, t_min, t_max, colors["phase"])

    if args.azadeh_binary:
        plot_spacetime_panel(
            fig,
            ax_rho,
            t_edges,
            y_edges,
            density_map,
            r"Local packing fraction",
            colors["packing_cmap"],
            r"$\phi(y,t)$",
            args.comm_time,
            colors["phase"],
            robust=True,
            force_zero_min=True,
            show_ylabel=True,
            center_value=float(np.nanmedian(density_map)),
        )
    else:
        plot_spacetime_panel(
            fig,
            ax_rho,
            t_edges,
            y_edges,
            density_map,
            r"Relative density field",
            "bwr",
            rf"$\rho/\rho_0 - 1$ with $\rho_0 = 1/{args.y_bins}$",
            args.comm_time,
            colors["phase"],
            robust=True,
            force_zero_min=False,
            show_ylabel=True,
            center_zero=True,
        )
    if args.azadeh_binary:
        plot_binary_weight_spacetime_panel(
            fig,
            ax_wmap,
            t_edges,
            y_edges,
            w_map,
            binary_policy_colors,
            args.comm_time,
            colors["phase"],
            show_ylabel=False,
        )
    else:
        plot_spacetime_panel(fig, ax_wmap, t_edges, y_edges, w_map, r"Weight field", colors["weight_cmap"], r"$\langle w \rangle$", args.comm_time, colors["phase"], robust=True, force_zero_min=False, show_ylabel=False)
    plot_spacetime_panel(fig, ax_qmap, t_edges, y_edges, q_map, r"Score field", colors["score_cmap"], r"$\langle q \rangle$", args.comm_time, colors["phase"], robust=True, force_zero_min=False, show_ylabel=False)

    add_panel_tag(ax_ctrl, "(a)")
    add_panel_tag(ax_shot0, "(b)")
    add_panel_tag(ax_shot1, "(c)")
    add_panel_tag(ax_shot2, "(d)")
    add_panel_tag(ax_frac, "(e)")
    add_panel_tag(ax_w, "(f)")
    add_panel_tag(ax_q, "(g)")
    add_panel_tag(ax_rho, "(h)")
    add_panel_tag(ax_wmap, "(i)")
    add_panel_tag(ax_qmap, "(j)")

    if args.azadeh_binary:
        suptitle = args.suptitle if args.suptitle else build_azadeh_binary_suptitle(
            args.experiment_id or args.video.stem,
        )
    elif args.azadeh:
        suptitle = args.suptitle if args.suptitle else build_azadeh_suptitle(
            args.experiment_id or args.video.stem,
        )
    else:
        suptitle = args.suptitle if args.suptitle else build_default_suptitle(
            args.controller,
            params,
            args.robust_max_runtime,
            experiment_id=args.experiment_id or args.video.stem,
        )
    fig.suptitle(suptitle, y=0.99 if azadeh_layout else 0.985, fontsize=13.5 if azadeh_layout else 18)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88 if azadeh_layout else 0.94, bottom=0.06)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    profile_fig = None
    if args.azadeh_binary:
        profile_fig = plot_azadeh_binary_profile_page(
            binary_profiles,
            binary_score_profiles,
            t_edges,
            profile_y_edges,
            args.comm_time,
            args.azadeh_binary_profile_metric,
            args.experiment_id or args.video.stem,
            time_cmap_name="turbo",
        )
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        if profile_fig is not None:
            pdf.savefig(profile_fig)

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        if profile_fig is not None:
            plt.close(profile_fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
