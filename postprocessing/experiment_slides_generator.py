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
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

LIGHT = 20
DARK = 80

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
    w_mean = (
        work.dropna(subset=["w"]).groupby(["t_idx", "y_bin"], sort=False)["w"]
        .mean().reindex(full_index).to_numpy().reshape(n_t, n_b)
    )
    q_mean = (
        work.dropna(subset=["q"]).groupby(["t_idx", "y_bin"], sort=False)["q"]
        .mean().reindex(full_index).to_numpy().reshape(n_t, n_b)
    )
    return t_edges, y_edges, rho_map, w_mean, q_mean


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
    """Extract title metadata from IDs such as ``alpha10-3_T1000_v2``."""
    if not experiment_id:
        return {}

    match = re.search(
        r"alpha(?P<alpha_base>\d+)-(?P<alpha_power>\d+)_T(?P<tq>\d+)"
        r"(?:_v(?P<version>\d+))?",
        Path(str(experiment_id)).stem,
    )
    if match is None:
        return {}

    # Experiment IDs use the convention ``alpha10-3`` == 10**(-3).
    # The ``10`` before the hyphen is part of that convention, not a
    # multiplicative coefficient.
    alpha_c = 10 ** (-int(match.group("alpha_power")))
    return {
        "t_q": int(match.group("tq")),
        "alpha_c": alpha_c,
        "version": match.group("version"),
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


def plot_spacetime_panel(fig, ax, t_edges, y_edges, data, title: str, cmap_name, cbar_label: str, comm_time: float, phase_color: str, robust: bool = False, force_zero_min: bool = False, show_ylabel: bool = False, center_zero: bool = False):
    vmin, vmax = choose_vrange(data, robust=robust)
    if force_zero_min:
        vmin = 0.0
    if center_zero:
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
    cmap = cmap_name.copy() if hasattr(cmap_name, "copy") else plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#F3F3F3")
    im = ax.pcolormesh(
        y_edges,
        t_edges / 60.0,
        np.ma.masked_invalid(data),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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
    params = build_controller_params(args)

    df = load_tracking_data(args.csv)
    frac = read_fraction_light_from_df(df, args.light_threshold, args.light_region)
    initial_weights = get_initial_weights(df)

    t_min_data = float(df["t"].min())
    t_max_data = float(df["t"].max())
    t_min = t_min_data if args.t_min is None else float(args.t_min)
    t_max = t_max_data if args.t_max is None else float(args.t_max)
    y_min = float(df["y"].min()) if args.y_min is None else float(args.y_min)
    y_max = float(df["y"].max()) if args.y_max is None else float(args.y_max)

    t_edges, y_edges, rho_map, w_map, q_map = build_maps_fast(
        df,
        y_bins=args.y_bins,
        y_min=y_min,
        y_max=y_max,
        t_bins=args.t_bins,
        t_min=t_min,
        t_max=t_max,
        t_bin_width=args.t_bin_width,
        t_edges=args.t_edges,
    )

    rho_rel_map, rho_0 = build_relative_density_map(rho_map, args.y_bins)

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

    plot_fraction_panel(ax_frac, frac, args.comm_time, t_min, t_max, colors["density_line"], colors["phase"])
    plot_population_traces(ax_w, df, "w", r"$w$", r"Weight trajectories and mean", colors["weight_line"], colors["weight_traces_cmap"], args.comm_time, t_min, t_max, colors["phase"])
    plot_population_traces(ax_q, df, "q", r"$q$", r"Score trajectories and mean", colors["score_line"], colors["score_traces_cmap"], args.comm_time, t_min, t_max, colors["phase"])

    plot_spacetime_panel(
        fig,
        ax_rho,
        t_edges,
        y_edges,
        rho_rel_map,
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

    suptitle = args.suptitle if args.suptitle else build_default_suptitle(
        args.controller,
        params,
        args.robust_max_runtime,
        experiment_id=args.experiment_id or args.video.stem,
    )
    fig.suptitle(suptitle, y=0.985, fontsize=18)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.06)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
