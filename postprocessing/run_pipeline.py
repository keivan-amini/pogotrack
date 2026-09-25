#!/usr/bin/env python3
"""Run the complete Pogotrack postprocessing workflow.

Configuration
-------------
The workflow is configured by ``postprocessing/pipeline.yaml``. Each entry in
``experiments`` describes one video, its robot count, optional dynamics-video
worker count, merge timing, and slide settings. Paths are resolved relative to
the repository root. Integer timing values can be supplied manually, or set to
``auto`` to detect them from the tracking and RGB-ID CSV files.

Modes
-----
``all`` (default)
    Run batch tracking, then ID merging, then slide generation. Tracking for
    different experiments runs concurrently as independent subprocesses.

``batch``
    Run standard tracking and RGB-ID tracking only. The two tracking outputs
    are written to each experiment's results directory together with logs.

``rgb``
    Run RGB-ID tracking only. This is useful when the standard tracking CSVs
    already exist and only ``RGB_ID.csv`` needs to be regenerated.

``merge``
    Use the standard CSV, RGB-ID CSV, and probe Feather file to create the
    merged dataset. Automatic ``seconds_difference`` and RGB start-frame
    detection are applied in this mode.

``slides``
    Generate the PNG/PDF experiment summary from an existing merged dataset.

Examples
--------
    # Run every enabled stage for every configured experiment.
    python postprocessing/run_pipeline.py

    # Run only one stage.
    python postprocessing/run_pipeline.py --stage batch
    python postprocessing/run_pipeline.py --stage rgb
    python postprocessing/run_pipeline.py --stage merge
    python postprocessing/run_pipeline.py --stage slides

    # Restrict a stage, or the complete workflow, to one experiment.
    python postprocessing/run_pipeline.py --stage merge --experiment alpha10-4_T100_v2
    python postprocessing/run_pipeline.py --experiment alpha10-4_T100_v2

    # Print the commands without running them.
    python postprocessing/run_pipeline.py --dry-run

Batch parallelism comes from independent tracking subprocesses controlled by
``batch.max_workers``. Individual dynamics videos can additionally use the
per-experiment ``workers`` value, which is passed to ``main.py`` as
``--workers``. RGB-ID tracking remains single-worker. During concurrent
tracking, each subprocess receives a stable tqdm terminal position and its
output is also written to the experiment-specific log file.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import pandas as pd


SLIDE_OPTION_FLAGS = {
    "light_threshold": "--light-threshold",
    "light_region": "--light-region",
    "comm_time": "--comm-time",
    "t_init_shot": "--t-init-shot",
    "t_mid_shot": "--t-mid-shot",
    "t_final_shot": "--t-final-shot",
    "y_bins": "--y-bins",
    "y_min": "--y-min",
    "y_max": "--y-max",
    "t_bins": "--t-bins",
    "t_bin_width": "--t-bin-width",
    "t_edges": "--t-edges",
    "t_min": "--t-min",
    "t_max": "--t-max",
    "robust_light_min": "--robust-light-min",
    "robust_light_max": "--robust-light-max",
    "robust_max_runtime": "--robust-max-runtime",
    "policy_run_time_plot": "--policy-run-time-plot",
    "tau_r_max_plot": "--tau-r-max-plot",
    "tau_r_0_plot": "--tau-r-0-plot",
    "sigma_policy": "--sigma-policy",
    "alpha": "--alpha",
    "delta": "--delta",
    "suptitle": "--suptitle",
}

SLIDE_BOOLEAN_FLAGS = {
    "no_tex": "--no-tex",
    "show": "--show",
}


_TERMINAL_OUTPUT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Experiment:
    identifier: str
    date: str
    video: str
    n_robots: int
    workers: int
    probe_file: str | None
    outputs: dict[str, str]
    merge: dict[str, Any]
    slides: dict[str, Any]


def load_config(config_path: Path) -> tuple[Path, dict[str, Any], list[Experiment]]:
    config_path = config_path.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("The postprocessing configuration must contain a YAML mapping.")

    project_root_value = config.get("project_root", "..")
    project_root = (config_path.parent / project_root_value).resolve()

    raw_experiments = config.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("Configuration must contain a non-empty `experiments` list.")

    experiments = []
    identifiers = set()
    for raw in raw_experiments:
        if not isinstance(raw, dict):
            raise ValueError("Every experiment entry must be a YAML mapping.")

        required = {"id", "date", "video", "n_robots"}
        missing = required - raw.keys()
        if missing:
            raise ValueError(
                f"Experiment entry is missing required keys: {sorted(missing)}"
            )

        identifier = str(raw["id"])
        if identifier in identifiers:
            raise ValueError(f"Duplicate experiment id: {identifier}")
        identifiers.add(identifier)

        n_robots = int(raw["n_robots"])
        if n_robots <= 0:
            raise ValueError(f"n_robots must be positive for {identifier}")

        workers = int(raw.get("workers", 1))
        if workers <= 0:
            raise ValueError(f"workers must be positive for {identifier}")

        experiments.append(
            Experiment(
                identifier=identifier,
                date=str(raw["date"]),
                video=str(raw["video"]),
                n_robots=n_robots,
                workers=workers,
                probe_file=(str(raw["probe_file"]) if raw.get("probe_file") else None),
                outputs={str(key): str(value) for key, value in (raw.get("outputs") or {}).items()},
                merge=dict(raw.get("merge") or {}),
                slides=dict(raw.get("slides") or {}),
            )
        )

    return project_root, config, experiments


def resolve_configured_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def output_name(experiment: Experiment, key: str, default: str) -> str:
    template = experiment.outputs.get(key, default)
    try:
        return template.format(
            id=experiment.identifier,
            date=experiment.date,
            video_stem=Path(experiment.video).stem,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder {{{exc.args[0]}}} in outputs.{key} "
            f"for {experiment.identifier}"
        ) from exc


def get_paths(project_root: Path, config: dict[str, Any], experiment: Experiment) -> dict[str, Path]:
    paths = config.get("paths") or {}
    data_root = resolve_configured_path(project_root, str(paths.get("data", "data")))
    results_root = resolve_configured_path(project_root, str(paths.get("results", "results")))
    tracking_config = resolve_configured_path(
        project_root, str(paths.get("tracking_config", "config/default.yaml"))
    )

    data_dir = data_root / experiment.date
    output_dir = results_root / experiment.date / experiment.identifier
    normal_video = data_dir / experiment.video
    rgb_video = data_dir / f"id_{experiment.video}"
    background = data_dir / "bkg.bmp"
    probe_file = data_dir / (experiment.probe_file or f"{experiment.identifier}.feather")

    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "normal_video": normal_video,
        "rgb_video": rgb_video,
        "background": background,
        "probe_file": probe_file,
        "tracking_config": tracking_config,
        "normal_output": output_dir / output_name(
            experiment, "normal", f"{experiment.identifier}.csv"
        ),
        "rgb_output": output_dir / output_name(
            experiment, "rgb_id", f"rgb_{experiment.identifier}.csv"
        ),
        "merged_output": output_dir / output_name(
            experiment, "merged", f"merged_{experiment.identifier}.csv"
        ),
        "slides_output": output_dir / output_name(
            experiment, "slides", experiment.identifier
        ),
    }


def configured_python(project_root: Path, config: dict[str, Any]) -> Path:
    value = config.get("python")
    if value:
        candidate = resolve_configured_path(project_root, str(value))
        if not candidate.exists():
            raise FileNotFoundError(f"Configured Python interpreter does not exist: {candidate}")
        return candidate
    return Path(sys.executable)


def check_inputs(paths: dict[str, Path], experiment: Experiment) -> None:
    required = (
        "normal_video",
        "rgb_video",
        "background",
        "probe_file",
        "tracking_config",
    )
    missing = [str(paths[key]) for key in required if not paths[key].is_file()]
    if missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing input files for {experiment.identifier}:\n  {formatted}")


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def run_command(
    command: list[str],
    *,
    project_root: Path,
    log_path: Path | None = None,
    dry_run: bool = False,
    progress_position: int | None = None,
    progress_desc: str | None = None,
) -> None:
    print(f"$ {command_text(command)}", flush=True)
    if dry_run:
        return

    environment = os.environ.copy()
    # Keep progress bars and ordinary print() calls visible immediately when
    # the child process writes to the pipe used by the live log tee below.
    environment["PYTHONUNBUFFERED"] = "1"
    if progress_position is not None:
        # The child owns the tqdm instance. Give every concurrently running
        # experiment a stable terminal row and let the bars disappear cleanly
        # when that command finishes.
        environment["POGOTRACK_TQDM_POSITION"] = str(progress_position)
        environment["POGOTRACK_TQDM_LEAVE"] = "0"
        if progress_desc:
            environment["POGOTRACK_TQDM_DESC"] = progress_desc

    if log_path is None:
        subprocess.run(
            command,
            cwd=project_root,
            check=True,
            env=environment,
        )
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        log.write(f"$ {command_text(command)}\n\n".encode("utf-8"))
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            bufsize=0,
        )
        assert process.stdout is not None
        try:
            while True:
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                # A lock prevents two concurrently running experiments from
                # interleaving bytes in the terminal. The logs remain separate.
                with _TERMINAL_OUTPUT_LOCK:
                    terminal = getattr(sys.stdout, "buffer", sys.stdout)
                    terminal.write(chunk)
                    terminal.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        finally:
            process.stdout.close()

        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def write_tracking_config(
    base_config: Path,
    output_dir: Path,
    n_robots: int,
    rgb_enabled: bool,
) -> Path:
    with base_config.open(encoding="utf-8") as handle:
        tracking_config = yaml.safe_load(handle) or {}
    if not isinstance(tracking_config, dict):
        raise ValueError(f"Tracking config must contain a YAML mapping: {base_config}")

    tracking_config["N_POGO"] = n_robots
    tracking_config["RGB_ID_ANALYSIS"] = rgb_enabled

    fd, raw_path = tempfile.mkstemp(
        prefix="tracking_",
        suffix=".yaml",
        dir=output_dir,
        text=True,
    )
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(tracking_config, handle, sort_keys=False)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def run_tracking_mode(
    project_root: Path,
    python: Path,
    config: dict[str, Any],
    experiment: Experiment,
    *,
    rgb_enabled: bool,
    video_key: str,
    output_key: str,
    log_name: str,
    dry_run: bool,
    progress_position: int | None = None,
) -> None:
    paths = get_paths(project_root, config, experiment)
    check_inputs(paths, experiment)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    temporary_config = None
    try:
        if dry_run:
            tracking_config = paths["tracking_config"]
        else:
            temporary_config = write_tracking_config(
                paths["tracking_config"],
                paths["output_dir"],
                experiment.n_robots,
                rgb_enabled,
            )
            tracking_config = temporary_config

        command = [
            str(python),
            "-m",
            "main",
            "--video",
            str(paths[video_key]),
            "--background",
            str(paths["background"]),
            "--output",
            str(paths[output_key]),
            "--config",
            str(tracking_config),
        ]
        if not rgb_enabled and experiment.workers > 1:
            command.extend(["--workers", str(experiment.workers)])
        run_command(
            command,
            project_root=project_root,
            log_path=paths["output_dir"] / log_name,
            dry_run=dry_run,
            progress_position=progress_position,
            progress_desc=(
                f"{experiment.identifier}: "
                f"{'RGB-ID tracking' if rgb_enabled else 'tracking'}"
            ),
        )
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)


def run_tracking(
    project_root: Path,
    python: Path,
    config: dict[str, Any],
    experiment: Experiment,
    *,
    dry_run: bool,
    progress_position: int | None = None,
) -> None:
    """Run normal tracking, followed by RGB-ID tracking."""
    run_tracking_mode(
        project_root,
        python,
        config,
        experiment,
        rgb_enabled=False,
        video_key="normal_video",
        output_key="normal_output",
        log_name="tracking.log",
        dry_run=dry_run,
        progress_position=progress_position,
    )
    run_tracking_mode(
        project_root,
        python,
        config,
        experiment,
        rgb_enabled=True,
        video_key="rgb_video",
        output_key="rgb_output",
        log_name="rgb_tracking.log",
        dry_run=dry_run,
        progress_position=progress_position,
    )


def run_rgb_tracking(
    project_root: Path,
    python: Path,
    config: dict[str, Any],
    experiment: Experiment,
    *,
    dry_run: bool,
    progress_position: int | None = None,
) -> None:
    """Run RGB-ID tracking without recomputing the normal tracking CSV."""
    run_tracking_mode(
        project_root,
        python,
        config,
        experiment,
        rgb_enabled=True,
        video_key="rgb_video",
        output_key="rgb_output",
        log_name="rgb_tracking.log",
        dry_run=dry_run,
        progress_position=progress_position,
    )


def load_motion_data(path: Path, analysis_seconds: float) -> pd.DataFrame:
    """Load only the beginning of a dynamics CSV for motion-onset detection."""
    header = pd.read_csv(path, nrows=0)
    required = {"time", "x", "y"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"Dynamics CSV is missing columns: {sorted(missing)}")

    identity_column = "particle" if "particle" in header.columns else "id"
    usecols = ["time", "x", "y", identity_column]
    chunks = []
    first_time = None

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        chunk = chunk.apply(pd.to_numeric, errors="coerce").dropna()
        if chunk.empty:
            continue

        if first_time is None:
            first_time = float(chunk["time"].min())
        cutoff = first_time + float(analysis_seconds)
        chunk_max_time = float(chunk["time"].max())
        chunk = chunk.loc[chunk["time"] <= cutoff].copy()
        if not chunk.empty:
            chunks.append(chunk)

        # The tracking output is time ordered, so no later chunk can contain
        # useful rows once the requested initial interval has been passed.
        if chunk_max_time >= cutoff:
            break

    if not chunks:
        raise ValueError(f"No valid rows found in the first {analysis_seconds} s of {path}")

    data = pd.concat(chunks, ignore_index=True)
    data = data.rename(columns={identity_column: "robot"})
    data["robot"] = data["robot"].astype(int)
    return data.sort_values(["robot", "time"]).reset_index(drop=True)


def detect_motion_start(
    path: Path,
    *,
    analysis_seconds: float = 10.0,
    baseline_seconds: float = 1.5,
    displacement_window_seconds: float = 0.25,
    threshold_k: float = 6.0,
    minimum_displacement: float = 0.35,
    min_valid_robot_fraction: float = 0.75,
    min_consecutive_windows: int = 3,
) -> float:
    """Estimate the video-to-experiment delay from population displacement.

    A short displacement is computed for every tracked robot. The median
    displacement across the swarm is compared with a robust baseline from the
    initial stationary interval. The first sustained population-wide rise is
    returned as the delay used by ``process_id_trits.py``.
    """
    if analysis_seconds <= 0 or baseline_seconds <= 0:
        raise ValueError("Motion analysis and baseline durations must be positive")
    if displacement_window_seconds <= 0:
        raise ValueError("displacement_window_seconds must be positive")
    if not 0 < min_valid_robot_fraction <= 1:
        raise ValueError("min_valid_robot_fraction must be in (0, 1]")
    if min_consecutive_windows <= 0:
        raise ValueError("min_consecutive_windows must be positive")

    data = load_motion_data(path, analysis_seconds)
    times = np.sort(data["time"].unique())
    time_diffs = np.diff(times)
    time_diffs = time_diffs[time_diffs > 0]
    if not len(time_diffs):
        raise ValueError(f"Cannot estimate sampling interval from {path}")

    dt = float(np.median(time_diffs))
    window_frames = max(1, int(round(displacement_window_seconds / dt)))
    grouped = data.groupby("robot", sort=False)
    data["previous_time"] = grouped["time"].shift(window_frames)
    data["previous_x"] = grouped["x"].shift(window_frames)
    data["previous_y"] = grouped["y"].shift(window_frames)
    data["displacement"] = np.hypot(
        data["x"] - data["previous_x"],
        data["y"] - data["previous_y"],
    )
    data["window_duration"] = data["time"] - data["previous_time"]
    data = data.loc[
        data["displacement"].notna()
        & (data["window_duration"] <= displacement_window_seconds * 1.5)
    ].copy()
    if data.empty:
        raise ValueError(f"Not enough valid tracking data for motion detection: {path}")

    n_robots = int(data["robot"].nunique())
    signal = (
        data.groupby("time")
        .agg(
            displacement=("displacement", "median"),
            valid_robots=("robot", "nunique"),
        )
        .sort_index()
    )
    signal["valid_fraction"] = signal["valid_robots"] / n_robots

    first_time = float(signal.index.min())
    baseline = signal.loc[
        signal.index <= first_time + baseline_seconds, "displacement"
    ].to_numpy(dtype=float)
    if len(baseline) < max(3, min_consecutive_windows):
        raise ValueError("The motion-detection baseline contains too few samples")

    baseline_median = float(np.median(baseline))
    baseline_mad = float(np.median(np.abs(baseline - baseline_median)))
    robust_sigma = 1.4826 * baseline_mad
    threshold = max(
        float(minimum_displacement),
        baseline_median + float(threshold_k) * max(robust_sigma, 1e-6),
    )

    candidates = (
        (signal["displacement"] > threshold)
        & (signal["valid_fraction"] >= min_valid_robot_fraction)
    )
    candidate_times = signal.index.to_numpy(dtype=float)
    run = 0
    for index, candidate in enumerate(candidates.to_numpy(dtype=bool)):
        run = run + 1 if candidate else 0
        if run >= min_consecutive_windows:
            start_index = index - min_consecutive_windows + 1
            detected = float(candidate_times[start_index])
            print(
                f"Automatic seconds_difference: {detected:.3f}s "
                f"(motion threshold {threshold:.3f}, {n_robots} robots)"
            )
            return detected

    raise ValueError(
        f"Could not detect sustained swarm motion in the first {analysis_seconds:g}s of {path}. "
        "Set merge.seconds_difference manually or adjust merge.auto settings."
    )

def run_merge(
    project_root: Path,
    python: Path,
    config: dict[str, Any],
    experiment: Experiment,
    *,
    dry_run: bool,
    progress_position: int | None = None,
) -> None:
    paths = get_paths(project_root, config, experiment)
    check_inputs(paths, experiment)
    merge = experiment.merge
    if "seconds_difference" not in merge:
        raise ValueError(f"Missing merge.seconds_difference for {experiment.identifier}")

    seconds_difference = merge["seconds_difference"]
    if isinstance(seconds_difference, str) and seconds_difference.lower() in {
        "auto",
        "automatic",
    }:
        auto = dict(merge.get("auto") or {})
        if dry_run:
            seconds_difference = "<automatic motion detection>"
        else:
            seconds_difference = detect_motion_start(
                paths["normal_output"],
                analysis_seconds=float(auto.get("analysis_seconds", 10.0)),
                baseline_seconds=float(auto.get("baseline_seconds", 1.5)),
                displacement_window_seconds=float(
                    auto.get("displacement_window_seconds", 0.25)
                ),
                threshold_k=float(auto.get("threshold_k", 6.0)),
                minimum_displacement=float(auto.get("minimum_displacement", 0.35)),
                min_valid_robot_fraction=float(
                    auto.get("min_valid_robot_fraction", 0.75)
                ),
                min_consecutive_windows=int(auto.get("min_consecutive_windows", 3)),
            )

    command = [
        str(python),
        str(project_root / "postprocessing" / "process_id_trits.py"),
        "--rgb-id-path",
        str(paths["rgb_output"]),
        "--probe-path",
        str(paths["probe_file"]),
        "--dyn-path",
        str(paths["normal_output"]),
        "--seconds-difference",
        str(seconds_difference),
        "--results",
        str(paths["merged_output"]),
    ]
    start_rgb_frame = merge.get("start_rgb_frame")
    if isinstance(start_rgb_frame, str) and start_rgb_frame.lower() in {
        "auto",
        "automatic",
    }:
        start_rgb_frame = None
    if start_rgb_frame is not None:
        command.extend(["--start-rgb-frame", str(merge["start_rgb_frame"])])

    command.extend(["--expected-n-robots", str(experiment.n_robots)])

    run_command(
        command,
        project_root=project_root,
        log_path=paths["output_dir"] / "merge.log",
        dry_run=dry_run,
    )


def run_slides(
    project_root: Path,
    python: Path,
    config: dict[str, Any],
    experiment: Experiment,
    *,
    dry_run: bool,
    progress_position: int | None = None,
) -> None:
    paths = get_paths(project_root, config, experiment)
    if not paths["merged_output"].is_file() and not dry_run:
        raise FileNotFoundError(f"Missing merged dataset: {paths['merged_output']}")

    slides = experiment.slides
    if not slides.get("enabled", True):
        print(f"Skipping slides for {experiment.identifier}: disabled in config")
        return

    controller = str(slides.get("controller", "tanh"))
    command = [
        str(python),
        str(project_root / "postprocessing" / "experiment_slides_generator.py"),
        "--csv",
        str(paths["merged_output"]),
        "--video",
        str(paths["normal_video"]),
        "--controller",
        controller,
        "--experiment-id",
        experiment.identifier,
        "--output",
        str(paths["slides_output"]),
    ]

    options = slides.get("options") or {}
    for key, value in options.items():
        if key in SLIDE_OPTION_FLAGS:
            if value is not None:
                command.extend([SLIDE_OPTION_FLAGS[key], str(value)])
        elif key in SLIDE_BOOLEAN_FLAGS and bool(value):
            command.append(SLIDE_BOOLEAN_FLAGS[key])
        else:
            valid_keys = sorted({*SLIDE_OPTION_FLAGS, *SLIDE_BOOLEAN_FLAGS})
            raise ValueError(
                f"Unknown slides.options key `{key}` for {experiment.identifier}. "
                f"Valid keys: {valid_keys}"
            )

    run_command(
        command,
        project_root=project_root,
        log_path=paths["output_dir"] / "slides.log",
        dry_run=dry_run,
    )


def run_stage(
    stage: str,
    project_root: Path,
    python: Path,
    config: dict[str, Any],
    experiments: list[Experiment],
    *,
    max_workers: int,
    dry_run: bool,
) -> None:
    if stage == "batch":
        function = run_tracking
    elif stage == "rgb":
        function = run_rgb_tracking
    elif stage == "merge":
        function = run_merge
    elif stage == "slides":
        function = run_slides
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if dry_run or max_workers <= 1 or len(experiments) <= 1:
        for experiment in experiments:
            print(f"\n[{stage}] {experiment.identifier}")
            function(
                project_root,
                python,
                config,
                experiment,
                dry_run=dry_run,
                progress_position=0,
            )
        return

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                function,
                project_root,
                python,
                config,
                experiment,
                dry_run=False,
                progress_position=index,
            ): experiment
            for index, experiment in enumerate(experiments)
        }
        for future in as_completed(futures):
            experiment = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((experiment.identifier, exc))
                print(f"[{stage}] FAILED: {experiment.identifier}: {exc}")
            else:
                print(f"[{stage}] completed: {experiment.identifier}")

    if failures:
        details = "\n".join(f"  {identifier}: {error}" for identifier, error in failures)
        raise RuntimeError(f"{stage} failed for {len(failures)} experiment(s):\n{details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("postprocessing/pipeline.yaml"),
        help="Postprocessing YAML configuration",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "batch", "rgb", "merge", "slides"],
        default="all",
        help="Run one stage or all stages in order",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="Run only this experiment id; may be supplied more than once",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root, config, experiments = load_config(args.config)
    python = configured_python(project_root, config)

    if args.experiment:
        selected = set(args.experiment)
        experiments = [experiment for experiment in experiments if experiment.identifier in selected]
        missing = selected - {experiment.identifier for experiment in experiments}
        if missing:
            raise ValueError(f"Unknown experiment id(s): {sorted(missing)}")

    batch_config = config.get("batch") or {}
    max_workers = int(batch_config.get("max_workers", 1))
    if max_workers <= 0:
        raise ValueError("batch.max_workers must be positive")

    stages = [args.stage] if args.stage != "all" else ["batch", "merge", "slides"]
    if args.stage in {"all", "batch"} and not bool(batch_config.get("enabled", True)):
        stages = [stage for stage in stages if stage != "batch"]
        print("Batch tracking is disabled in the configuration.")

    print(f"Project root: {project_root}")
    print(f"Python: {python}")
    print(f"Experiments: {', '.join(experiment.identifier for experiment in experiments)}")

    for stage in stages:
        run_stage(
            stage,
            project_root,
            python,
            config,
            experiments,
            max_workers=max_workers if stage in {"batch", "rgb"} else 1,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
