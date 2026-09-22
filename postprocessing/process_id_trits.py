'''

Example:

/usr/bin/python3 process_id_trits.py \
  --rgb-id-path results/22-04-26/RGB_ID_t15.csv \
  --probe-path data/22-04-26/score_t15.feather \
  --dyn-path results/22-04-26/score_t15.csv \
  --seconds-difference 0.5 \
  --start-rgb-frame 32 \
  --results results/22-04-26/merged_t15.csv



'''





from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ============================================================
# I/O
# ============================================================

def load_rgb_id_data(rgb_id_experiment_path: str) -> pd.DataFrame:
    """
    Load the RGB-ID CSV produced by process_rgb_id().

    Expected columns:
        frame, time, id, x, y, theta, led_x, led_y, R, G, B

    Internally, temporary video labels are renamed from `id` to `temp_id`
    to avoid confusion with the final real robot ID.
    """
    path = Path(rgb_id_experiment_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing RGB-ID CSV: {path}")

    df = pd.read_csv(path)

    required = {
        "frame", "time", "id", "x", "y", "theta", "led_x", "led_y", "R", "G", "B"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RGB-ID CSV is missing columns: {sorted(missing)}")

    df = df.copy()
    df = df.rename(columns={"id": "temp_id"})

    numeric_cols = [
        "frame", "time", "temp_id", "x", "y", "theta", "led_x", "led_y", "R", "G", "B"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()
    df["frame"] = df["frame"].astype(int)
    df["temp_id"] = df["temp_id"].astype(int)

    return df.sort_values(["frame", "temp_id"]).reset_index(drop=True)


def load_probe_codes(probe_file_path: str, n_trits: int = 11) -> pd.DataFrame:
    """
    Load unique real robot IDs from the Feather file and convert each one
    to an 11-trit ternary code.
    """
    path = Path(probe_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing probe file: {path}")

    df_probe = pd.read_feather(path)
    if "robot" not in df_probe.columns:
        raise ValueError("Probe Feather file must contain a `robot` column.")

    real_ids = sorted(df_probe["robot"].dropna().astype(int).unique().tolist())

    return pd.DataFrame(
        {
            "real_id": real_ids,
            "trits": [int_to_base3_fixed_width(robot_id, width=n_trits) for robot_id in real_ids],
        }
    )


# ============================================================
# Utilities
# ============================================================

def estimate_fps(df: pd.DataFrame) -> float:
    """
    Estimate FPS from unique (frame, time) pairs.
    """
    ft = df[["frame", "time"]].drop_duplicates().sort_values("frame")
    dframe = np.diff(ft["frame"].to_numpy(dtype=float))
    dtime = np.diff(ft["time"].to_numpy(dtype=float))

    valid = dtime > 0
    if not np.any(valid):
        raise ValueError("Cannot estimate FPS from frame/time columns.")

    fps = np.median(dframe[valid] / dtime[valid])
    if fps <= 0:
        raise ValueError(f"Estimated FPS is invalid: {fps}")

    return float(fps)


def int_to_base3_fixed_width(value: int, width: int = 11) -> str:
    """
    Convert a non-negative integer to base 3 and left-pad with zeros to `width`.
    """
    if value < 0:
        raise ValueError("Robot ID must be non-negative.")

    out = [0] * width
    n = int(value)

    for i in range(width - 1, -1, -1):
        out[i] = n % 3
        n //= 3

    if n != 0:
        raise ValueError(
            f"Robot ID {value} does not fit in {width} base-3 digits."
        )

    return "".join(map(str, out))


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hamming distance requires equal-length strings.")
    return int(sum(c1 != c2 for c1, c2 in zip(a, b)))


# ============================================================
# Start-frame detection
# ============================================================

def build_frame_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one per-frame signal summary to detect the common LED onset.

    We use the per-robot max(R,G,B), then aggregate per frame with a median.
    """
    tmp = df[["frame", "time", "R", "G", "B"]].copy()
    tmp["rgb_max"] = tmp[["R", "G", "B"]].max(axis=1)

    signal_df = (
        tmp.groupby("frame", as_index=False)
        .agg(
            time=("time", "median"),
            signal=("rgb_max", "median"),
            R_med=("R", "median"),
            G_med=("G", "median"),
            B_med=("B", "median"),
        )
        .sort_values("frame")
        .reset_index(drop=True)
    )
    return signal_df


def detect_experiment_start(
    df: pd.DataFrame,
    fps: float,
    baseline_seconds: float = 1.5,
    threshold_k: float = 6.0,
    min_consecutive_frames: int = 3,
    min_active_fraction: float = 0.75,
) -> tuple[int, pd.DataFrame]:
    """
    Detect the common start frame where LEDs first turn on.

    Strategy:
    - compute per-frame median signal = median over robots of max(R,G,B)
    - estimate a robust baseline from the first baseline_seconds
    - require the threshold crossing in most robots
    - detect the first run of min_consecutive_frames above threshold

    If no clear onset is found, fall back to the first frame.
    """
    signal_df = build_frame_signal(df)

    n_baseline = max(5, int(round(baseline_seconds * fps)))
    n_baseline = min(n_baseline, max(5, len(signal_df) // 4))

    baseline = signal_df.iloc[:n_baseline]["signal"].to_numpy(dtype=float)
    base_med = float(np.median(baseline))
    base_mad = float(np.median(np.abs(baseline - base_med)))
    robust_sigma = 1.4826 * base_mad
    threshold = base_med + threshold_k * max(robust_sigma, 1.0)

    if not 0 < min_active_fraction <= 1:
        raise ValueError("min_active_fraction must be in (0, 1].")

    active_fraction = (
        df.assign(rgb_max=df[["R", "G", "B"]].max(axis=1))
        .assign(active=lambda work: work["rgb_max"] > threshold)
        .groupby("frame")["active"]
        .mean()
        .reindex(signal_df["frame"])
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    signal_df["active_fraction"] = active_fraction
    active = (
        (signal_df["signal"].to_numpy(dtype=float) > threshold)
        & (active_fraction >= min_active_fraction)
    )

    start_frame = None
    run = 0
    for i, is_active in enumerate(active):
        run = run + 1 if is_active else 0
        if run >= min_consecutive_frames:
            start_frame = int(signal_df.iloc[i - min_consecutive_frames + 1]["frame"])
            break

    if start_frame is None:
        start_frame = int(signal_df["frame"].min())
        detected = False
    else:
        detected = True

    signal_df = signal_df.copy()
    signal_df["threshold"] = threshold
    signal_df["is_active"] = active
    signal_df["detected_start_frame"] = start_frame
    signal_df["start_detected_from_signal"] = detected

    return start_frame, signal_df


# ============================================================
# Windowing and robust RGB summaries
# ============================================================

def build_trit_windows(
    start_frame: int,
    n_trits: int,
    flash_ms: int,
    blank_ms: int,
    fps: float,
    trim_frames: int = 2,
) -> list[dict]:
    """
    Build half-open windows [start, end) for each trit burst, and
    trimmed windows [keep_start, keep_end) used for median RGB aggregation.
    """
    flash_frames = int(round((flash_ms / 1000.0) * fps))
    blank_frames = int(round((blank_ms / 1000.0) * fps))

    if flash_frames <= 0:
        raise ValueError("flash_frames must be positive.")
    if trim_frames * 2 >= flash_frames:
        raise ValueError(
            f"trim_frames={trim_frames} is too large for flash_frames={flash_frames}."
        )

    windows = []
    step = flash_frames + blank_frames

    for trit_pos in range(n_trits):
        start = int(start_frame + trit_pos * step)
        end = int(start + flash_frames)
        keep_start = int(start + trim_frames)
        keep_end = int(end - trim_frames)

        windows.append(
            {
                "trit_pos": trit_pos,
                "frame_start": start,
                "frame_end": end,
                "keep_start": keep_start,
                "keep_end": keep_end,
            }
        )

    return windows


def aggregate_trit_windows(
    df: pd.DataFrame,
    windows: list[dict],
    expected_n_robots: int = None,
) -> pd.DataFrame:
    """
    For each temporary robot and each trit position, aggregate one robust
    RGB triplet using the median over the trimmed window.
    """
    temp_ids = sorted(df["temp_id"].unique().tolist())
    n_robots = len(temp_ids)

    if expected_n_robots is not None and n_robots != int(expected_n_robots):
        raise ValueError(
            f"Found {n_robots} robots in RGB-ID CSV, expected {expected_n_robots}."
        )

    rows = []

    for win in windows:
        w = df.loc[
            (df["frame"] >= win["keep_start"]) & (df["frame"] < win["keep_end"])
        ].copy()

        if w.empty:
            raise ValueError(
                f"No data in trimmed window for trit {win['trit_pos']} "
                f"({win['keep_start']}:{win['keep_end']})."
            )

        grouped = (
            w.groupby("temp_id", as_index=False)
            .agg(
                x=("x", "median"),
                y=("y", "median"),
                theta=("theta", "median"),
                led_x=("led_x", "median"),
                led_y=("led_y", "median"),
                R=("R", "median"),
                G=("G", "median"),
                B=("B", "median"),
                n_samples=("frame", "count"),
            )
            .sort_values("temp_id")
            .reset_index(drop=True)
        )

        if len(grouped) != n_robots:
            missing = sorted(set(temp_ids) - set(grouped["temp_id"].tolist()))
            raise ValueError(
                f"Trit {win['trit_pos']} contains {len(grouped)} robots, expected {n_robots}. "
                f"Missing temp_id(s): {missing}"
            )

        grouped["trit_pos"] = int(win["trit_pos"])
        grouped["frame_start"] = int(win["frame_start"])
        grouped["frame_end"] = int(win["frame_end"])
        grouped["keep_start"] = int(win["keep_start"])
        grouped["keep_end"] = int(win["keep_end"])
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Trit decoding
# ============================================================

def decode_one_trit(R: float, G: float, B: float) -> tuple[int, str, float]:
    """
    Decode one trit with the simple argmax rule:
        R > others -> 0
        G > others -> 1
        B > others -> 2

    Returns:
        decoded_trit, dominant_channel, confidence_margin
    """
    values = np.array([float(R), float(G), float(B)], dtype=float)
    order = np.argsort(values)
    winner = int(order[-1])
    second = int(order[-2])
    confidence = float(values[winner] - values[second])

    channel_names = ["R", "G", "B"]
    dominant_channel = channel_names[winner]

    return winner, dominant_channel, confidence


def decode_trits(df_trits: pd.DataFrame, n_trits: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Decode all per-window RGB summaries into ternary sequences.
    """
    rows = []

    for row in df_trits.itertuples(index=False):
        decoded_trit, dominant_channel, confidence = decode_one_trit(row.R, row.G, row.B)

        rows.append(
            {
                "temp_id": int(row.temp_id),
                "trit_pos": int(row.trit_pos),
                "frame_start": int(row.frame_start),
                "frame_end": int(row.frame_end),
                "keep_start": int(row.keep_start),
                "keep_end": int(row.keep_end),
                "n_samples": int(row.n_samples),
                "x": float(row.x),
                "y": float(row.y),
                "theta": float(row.theta),
                "led_x": float(row.led_x),
                "led_y": float(row.led_y),
                "R": float(row.R),
                "G": float(row.G),
                "B": float(row.B),
                "decoded_trit": int(decoded_trit),
                "dominant_channel": dominant_channel,
                "confidence": float(confidence),
            }
        )

    decoded_df = pd.DataFrame(rows).sort_values(["temp_id", "trit_pos"]).reset_index(drop=True)

    counts = decoded_df.groupby("temp_id")["trit_pos"].nunique()
    bad = counts[counts != n_trits]
    if not bad.empty:
        raise ValueError(
            "Some robots do not have exactly "
            f"{n_trits} decoded trits: {bad.to_dict()}"
        )

    seq_df = (
        decoded_df.sort_values(["temp_id", "trit_pos"])
        .groupby("temp_id", as_index=False)
        .agg(
            decoded_trits=("decoded_trit", lambda s: "".join(map(str, s.tolist()))),
            mean_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
        )
    )

    return decoded_df, seq_df


# ============================================================
# Matching to real robot IDs
# ============================================================

def assign_real_ids(
    seq_df: pd.DataFrame,
    probe_codes: pd.DataFrame,
) -> pd.DataFrame:
    """
    One-to-one assignment between decoded trit sequences and real robot IDs
    using Hamming distance and Hungarian matching.

    Supports:
    - decoded robots >= probe IDs   : allowed
    - decoded robots <  probe IDs   : error

    Unmatched decoded robots are kept with NaN real_id.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError(
            "scipy is required for ID assignment. Install it with: pip install scipy"
        ) from e

    decoded = seq_df.copy().sort_values("temp_id").reset_index(drop=True)
    probe = probe_codes.copy().sort_values("real_id").reset_index(drop=True)

    n_decoded = len(decoded)
    n_probe = len(probe)

    if n_decoded < n_probe:
        raise ValueError(
            f"Decoded {n_decoded} robots but Feather contains {n_probe} unique robot IDs. "
            "This case is not allowed because some real IDs would have no candidate match."
        )

    cost = np.zeros((n_decoded, n_probe), dtype=int)

    for i, dec_code in enumerate(decoded["decoded_trits"]):
        for j, true_code in enumerate(probe["trits"]):
            cost[i, j] = hamming_distance(dec_code, true_code)

    # Rectangular assignment: when n_decoded > n_probe, only n_probe rows are matched
    row_ind, col_ind = linear_sum_assignment(cost)

    assigned = decoded.copy()
    assigned["real_id"] = pd.NA
    assigned["real_trits"] = pd.NA
    assigned["hamming_distance"] = pd.NA
    assigned["matched"] = False

    for i, j in zip(row_ind, col_ind):
        assigned.loc[i, "real_id"] = int(probe.loc[j, "real_id"])
        assigned.loc[i, "real_trits"] = str(probe.loc[j, "trits"])
        assigned.loc[i, "hamming_distance"] = int(cost[i, j])
        assigned.loc[i, "matched"] = True

    return assigned.sort_values("temp_id").reset_index(drop=True)

def attach_real_ids_per_trit(
    decoded_df: pd.DataFrame,
    assigned_ids: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the matched real ID to each per-trit decoded row.
    """
    out = decoded_df.merge(
        assigned_ids[["temp_id", "real_id", "real_trits", "hamming_distance", "matched"]],
        on="temp_id",
        how="left",
        validate="many_to_one",
    )

    def _assigned_trit(row):
        if pd.isna(row["real_trits"]):
            return pd.NA
        return int(str(row["real_trits"])[int(row["trit_pos"])])

    out["assigned_trit"] = out.apply(_assigned_trit, axis=1)
    return out


# ============================================================
# Final pose-ID dataset
# ============================================================

def build_pose_id_dataset(df_raw: pd.DataFrame, assigned_ids: pd.DataFrame) -> pd.DataFrame:
    """
    Build final dataset:
        x0, y0, theta0, id
    where `id` is the matched real robot ID.
    Only matched robots are kept.
    """
    pose0 = (
        df_raw.sort_values(["temp_id", "frame"])
        .groupby("temp_id", as_index=False)
        .first()[["temp_id", "x", "y", "theta"]]
        .rename(columns={"x": "x0", "y": "y0", "theta": "theta0"})
    )

    matched_ids = assigned_ids.loc[assigned_ids["matched"]].copy()

    out = pose0.merge(
        matched_ids[["temp_id", "real_id"]],
        on="temp_id",
        how="inner",
        validate="one_to_one",
    )

    out = out.rename(columns={"real_id": "id"})
    return out[["x0", "y0", "theta0", "id"]].sort_values("id").reset_index(drop=True)

# ============================================================
# Full RGB-ID processing
# ============================================================

def process_rgb_id_experiment(
    rgb_id_experiment_path: str,
    probe_file_path: str,
    n_trits: int = 11,
    fps: float = None,
    flash_ms: int = 1000,
    blank_ms: int = 0,
    trim_frames: int = 2,
    start_frame: int = None,
    baseline_seconds: float = 1.5,
    threshold_k: float = 6.0,
    min_consecutive_frames: int = 3,
    min_active_fraction: float = 0.75,
    expected_n_robots: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline for the new ternary RGB-ID experiment.

    Returns:
        df_raw            : raw RGB-ID CSV data
        signal_df         : per-frame signal used for start detection
        decoded_trits_df  : one row per temp_id per trit
        assigned_ids_df   : one row per temp_id with matched real ID
        pose_id_df        : x0, y0, theta0, id
    """
    df_raw = load_rgb_id_data(rgb_id_experiment_path)

    if fps is None:
        fps = estimate_fps(df_raw)

    if start_frame is None:
        start_frame, signal_df = detect_experiment_start(
            df_raw,
            fps=fps,
            baseline_seconds=baseline_seconds,
            threshold_k=threshold_k,
            min_consecutive_frames=min_consecutive_frames,
            min_active_fraction=min_active_fraction,
        )
    else:
        signal_df = build_frame_signal(df_raw)
        signal_df["threshold"] = np.nan
        signal_df["is_active"] = np.nan
        signal_df["detected_start_frame"] = int(start_frame)
        signal_df["start_detected_from_signal"] = False

    windows = build_trit_windows(
        start_frame=start_frame,
        n_trits=n_trits,
        flash_ms=flash_ms,
        blank_ms=blank_ms,
        fps=fps,
        trim_frames=trim_frames,
    )

    df_windowed = aggregate_trit_windows(
        df_raw,
        windows=windows,
        expected_n_robots=expected_n_robots,
    )

    decoded_trits_df, seq_df = decode_trits(df_windowed, n_trits=n_trits)
    probe_codes = load_probe_codes(probe_file_path, n_trits=n_trits)
    assigned_ids_df = assign_real_ids(seq_df, probe_codes)
    decoded_trits_df = attach_real_ids_per_trit(decoded_trits_df, assigned_ids_df)
    pose_id_df = build_pose_id_dataset(df_raw, assigned_ids_df)

    return df_raw, signal_df, decoded_trits_df, assigned_ids_df, pose_id_df


# ============================================================
# Dynamic merge
# ============================================================

def _load_table(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".feather":
        return pd.read_feather(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _assign_particles_to_ids(first_dyn: pd.DataFrame, pose_id: pd.DataFrame) -> pd.DataFrame:
    """
    Assign dynamic particles to reconstructed robot IDs using nearest initial
    (x, y) and a one-to-one assignment.

    If there are more particles than IDs, only the best matching subset is kept.
    Extra particles are ignored.
    """
    if len(first_dyn) < len(pose_id):
        raise ValueError(
            f"First dynamic frame has only {len(first_dyn)} particles, "
            f"but pose_id has {len(pose_id)} rows, so not all IDs can be assigned."
        )

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError(
            "scipy is required for particle-to-id assignment. Install it with: pip install scipy"
        ) from e

    dyn_xy = first_dyn[["x", "y"]].to_numpy(dtype=float)
    ref_xy = pose_id[["x0", "y0"]].to_numpy(dtype=float)

    cost_matrix = np.linalg.norm(dyn_xy[:, None, :] - ref_xy[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    out = first_dyn.iloc[row_ind][["particle"]].copy().reset_index(drop=True)
    out["id"] = pose_id.iloc[col_ind]["id"].to_numpy()
    out["match_distance"] = cost_matrix[row_ind, col_ind]

    return out.sort_values("particle").reset_index(drop=True)


def merge_datasets(
    dynamical_path="results/rgb_id_dyn.csv",
    probe_data="data/08-04-26/rgb_id_finetuning.feather",
    pose_id_data="results/pose_id.csv",
    delay_s: float = 6.0,
    tolerance_s: float  = None,
) -> pd.DataFrame:
    """
    Merge:
    1) dynamics dataset: {time, x, y, theta, particle}
    2) probe dataset: {robot, chunk, i, t_s, q, w}
    3) reconstructed pose-id dataset: {x0, y0, theta0, id}

    Returns:
        DataFrame with columns {t, x, y, theta, q, w, id}
    """
    dyn = _load_table(dynamical_path)
    probe = _load_table(probe_data)
    pose_id = _load_table(pose_id_data)

    required_dyn = {"time", "x", "y", "theta", "particle"}
    required_probe = {"robot", "t_s", "q", "w"}
    required_pose = {"x0", "y0", "theta0", "id"}

    missing_dyn = required_dyn - set(dyn.columns)
    missing_probe = required_probe - set(probe.columns)
    missing_pose = required_pose - set(pose_id.columns)

    if missing_dyn:
        raise ValueError(f"Dynamic dataset is missing columns: {sorted(missing_dyn)}")
    if missing_probe:
        raise ValueError(f"Probe dataset is missing columns: {sorted(missing_probe)}")
    if missing_pose:
        raise ValueError(f"Pose-ID dataset is missing columns: {sorted(missing_pose)}")

    dyn = dyn.copy()
    probe = probe.copy()
    pose_id = pose_id.copy()

    dyn["time"] = pd.to_numeric(dyn["time"], errors="coerce")
    dyn["x"] = pd.to_numeric(dyn["x"], errors="coerce")
    dyn["y"] = pd.to_numeric(dyn["y"], errors="coerce")
    dyn["theta"] = pd.to_numeric(dyn["theta"], errors="coerce")
    dyn["particle"] = pd.to_numeric(dyn["particle"], errors="coerce")
    dyn = dyn.dropna(subset=["time", "x", "y", "theta", "particle"]).copy()
    dyn["particle"] = dyn["particle"].astype(int)

    dyn["t"] = dyn["time"].astype(float) - float(delay_s)
    dyn = dyn.loc[dyn["t"] >= 0].copy()

    if dyn.empty:
        raise ValueError("No dynamic rows remain after applying the delay filter.")

    first_t = float(dyn["t"].min())
    first_dyn = (
        dyn.loc[np.isclose(dyn["t"], first_t), ["particle", "x", "y", "theta"]]
        .sort_values("particle")
        .reset_index(drop=True)
    )

    particle_to_id = _assign_particles_to_ids(first_dyn=first_dyn, pose_id=pose_id)

    dyn = dyn.merge(
        particle_to_id[["particle", "id"]],
        on="particle",
        how="left",
        validate="many_to_one",
    )
    dyn["id"] = pd.to_numeric(dyn["id"], errors="coerce").astype("Int64")
    dyn["t"] = dyn["t"].astype(float)

    probe["id"] = pd.to_numeric(probe["robot"], errors="coerce").astype("Int64")
    probe["t_s"] = pd.to_numeric(probe["t_s"], errors="coerce").astype(float)
    probe["q"] = pd.to_numeric(probe["q"], errors="coerce")
    probe["w"] = pd.to_numeric(probe["w"], errors="coerce")
    probe = probe.dropna(subset=["id", "t_s"]).copy()

    merged_parts = []

    for robot_id, dyn_g in dyn.groupby("id", sort=False):
        dyn_g = dyn_g.sort_values("t").reset_index(drop=True)
        probe_g = (
            probe.loc[probe["id"] == robot_id, ["t_s", "q", "w"]]
            .sort_values("t_s")
            .reset_index(drop=True)
        )

        if probe_g.empty:
            dyn_g["q"] = np.nan
            dyn_g["w"] = np.nan
            merged_parts.append(dyn_g)
            continue

        merged_g = pd.merge_asof(
            dyn_g,
            probe_g,
            left_on="t",
            right_on="t_s",
            direction="nearest",
            tolerance=tolerance_s,
        )
        merged_parts.append(merged_g)

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values(["id", "t"]).reset_index(drop=True)

    return merged[["t", "x", "y", "theta", "q", "w", "id"]].copy()


# ============================================================
# Main
# ============================================================

#### SET PARAMETERS

N_TRITS = 11
FPS = 20.0
FLASH_MS = 1000
BLANK_MS = 0
TRIM_FRAMES = 3

BASELINE_SECONDS = 1.5
THRESHOLD_K = 6.0
MIN_CONSECUTIVE_FRAMES = 3
MIN_ACTIVE_FRACTION = 0.75
EXPECTED_N_ROBOTS = None

def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode RGB-ID trits, assign real robot IDs, and merge with dynamics."
    )

    parser.add_argument(
        "--rgb-id-path",
        type=Path,
        required=True,
        help="Path to the RGB_ID.csv file",
    )
    parser.add_argument(
        "--probe-path",
        type=Path,
        required=True,
        help="Path to the probe Feather/CSV file",
    )
    parser.add_argument(
        "--dyn-path",
        type=Path,
        required=True,
        help="Path to the dynamics CSV/Feather file",
    )
    parser.add_argument(
        "--seconds-difference",
        type=float,
        required=True,
        help="Delay in seconds used in merge_datasets(delay_s=...)",
    )
    parser.add_argument(
        "--start-rgb-frame",
        type=int,
        default=None,
        help="Start frame of RGB flashing sequence; omit to use auto-detection",
    )
    parser.add_argument(
        "--expected-n-robots",
        type=int,
        default=None,
        help="Expected number of robots in the RGB-ID CSV",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Output CSV path for the final merged dataset",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rgb_id_experiment_path = args.rgb_id_path
    probe_file_path = args.probe_path
    dynamical_path = args.dyn_path
    threshold_seconds = args.seconds_difference
    start_rgb_frame = args.start_rgb_frame
    merged_df_path = args.results

    out_dir = rgb_id_experiment_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_df_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw, df_signal, df_decoded, df_ids, df_pose_id = process_rgb_id_experiment(
        rgb_id_experiment_path=rgb_id_experiment_path,
        probe_file_path=probe_file_path,
        n_trits=N_TRITS,
        fps=FPS,
        flash_ms=FLASH_MS,
        blank_ms=BLANK_MS,
        trim_frames=TRIM_FRAMES,
        start_frame=start_rgb_frame,
        baseline_seconds=BASELINE_SECONDS,
        threshold_k=THRESHOLD_K,
        min_consecutive_frames=MIN_CONSECUTIVE_FRAMES,
        min_active_fraction=MIN_ACTIVE_FRACTION,
        expected_n_robots=(
            args.expected_n_robots
            if args.expected_n_robots is not None
            else EXPECTED_N_ROBOTS
        ),
    )

    # Useful for debuggging
    #df_signal.to_csv(out_dir / "score_calib_rgb_start_signal.csv", index=False)
    #df_decoded.to_csv(out_dir / "score_calib_decoded_trits.csv", index=False)
    #df_ids.to_csv(out_dir / "score_calib_decoded_ids.csv", index=False)
    df_pose_id.to_csv(out_dir / "score_calib_pose_id.csv", index=False)

    print("\nAssignment summary:")
    print(df_ids[["temp_id", "real_id", "decoded_trits", "real_trits", "hamming_distance", "matched"]])

    print("\nMatched robots:", int(df_ids["matched"].sum()))
    print("Unmatched decoded robots:", int((~df_ids["matched"]).sum()))
    print("\nDetected start frame:", int(df_signal["detected_start_frame"].iloc[0]))
    print("Detected from signal:", bool(df_signal["start_detected_from_signal"].iloc[0]))

    print("\nAssigned IDs:")
    print(df_ids)

    print("\nPose-ID dataset:")
    print(df_pose_id)

    df_final = merge_datasets(
        dynamical_path=dynamical_path,
        probe_data=probe_file_path,
        pose_id_data=out_dir / "score_calib_pose_id.csv",
        delay_s=threshold_seconds,
        tolerance_s=None,
    )
    df_final.to_csv(merged_df_path, index=False)

    print("\nFinal merged dataset:")
    print(df_final.head())
    print(f"\nSaved final merged dataset to: {merged_df_path}")


if __name__ == "__main__":
    main()
