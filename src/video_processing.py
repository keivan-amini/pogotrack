"""
@author: Keivan Amini, PhD student

pogotrack core module

This module has the aim to construct the VideoProcessing class,
useful to initialize and run the whole pogotrack pipeline.
It imports helpers and utility functions from the other modules
contained in src, and create a default config dictionary 
containing all the video processing parameters. Note: these
default parameters are overwritten by config/default.yaml.
"""

import time
import cv2
cv2.setNumThreads(1)
import multiprocessing as mp
import tempfile
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

from src.utils import (
    get_difference,
    binarize,
    find_contours,
    get_position,
    get_all_angles,
    track_objects,
    convert_datas,
    to_blue_channel,
    create_line_region_masks,
    filter_small_components,
    filter_small_components_in_rois,
    detect_hough_circles,
    deduplicate_circles,
    process_phototaxis_frame,
    detect_circles_with_fallback,
    circles_to_centers,
    compute_led_positions,
)

from src.plot_helpers import (
    debug_frame,
    visualize_contours,
    plot_trajectories,
    save_rgb_arena_debug_view,
    save_led_debug_gif,
    _measure_frame_rows,
    save_led_contact_sheet
)


def _progress_log(message):
    """Write a message without disturbing active tqdm progress bars."""
    tqdm.write(str(message))


def _process_video_chunk_worker(job):
    """Process one video chunk in a spawned worker and write raw detections."""
    (
        video_path,
        background_path,
        save_path,
        config_path,
        frame_start,
        frame_end,
        warmup_start,
        worker_index,
        worker_count,
    ) = job

    processor = VideoProcessor(
        video_path=video_path,
        background_path=background_path,
        save_path=save_path,
        config_path=config_path,
    )
    detections = processor.process(
        frame_start=frame_start,
        frame_end=frame_end,
        warmup_start=warmup_start,
        finalize=False,
        show_progress=True,
        progress_position=worker_index,
        progress_desc=f"Worker {worker_index + 1}/{worker_count}",
    )
    detections.to_csv(save_path, index=False)
    return save_path



class VideoProcessor:

    """
    Description
    -----------
    This class wraps-up all the video processing function
    and loops thorugh the input pogobot experiment video
    frame by frame. In this process, it extracts physical
    features of each pogobot (such as position and orientation)
    and their IDs.
    The final output is a .csv file, containing, respectively:

    |  time |  x  |  y  | theta | particle |
    |-------|-----|-----|-------|----------|
    |  0.0  |  78 |  60 |  18.1 |    0     |
    |  0.04 |  77 |  60 |  18.3 |    0     |
    |  ...  | ... | ... |  ...  |   ...    |
    
    where:

    - time: experiment time [s]
    - x: pogobots' horizontal coordinate [cm]
    - y: pogobots' vertical coordinate [cm]
    - theta: direction of the arrow in pogobots' head [°]
    - particle: pogobots' ID

    Methods
    -------
        _load_config(yaml_config:)
            merge default parameters with .yaml configuration.
        _load_video_and_background()
            load the input video and background image.
        _create_mask()
            create circular binary mask for the arena.
        process()
            run the complete video processing pipeline.

    Attributes
    ----------
        video_path (str):
            path to the input video file (.mp4).
        background_path (str):
            path to the background image file (.bmp).
        save_path (str):
            path where the processed output .csv will be saved.
        config (dict):
            dictionary containing processing parameters,
            loaded from defaults and .yaml.
        frames_to_visualize (set[int]):
            set of frame indices for which contour visualization
            will be generated.
        df (pd.DataFrame):
            dataframe storing intermediate tracking results.
        video (cv2.VideoCapture):
            OpenCV video capture object.
        background (np.ndarray):
            background image array.

    """

    DEFAULTS = { # For a complete parameters description, check config/default.yaml
        "N_POGO": 1,
        "THRESHOLD": 7,
        "AREAS": [6000, 7500],
        "PERIMETERS": [300, 600],
        "CENTER": [1512, 1531],
        "RADIUS": 1512,
        "RECT_MASK": False,
        "WIDTH": 10,
        "HEIGHT": 10,
        "SEARCH_RANGE": 200,
        "MEMORY": 5,
        "CENTROIDS_SIZE": 7,
        "ARROW_LENGTH_FRAME": 30,
        "TIP_LENGTH": 0.2,
        "FPS": 22.46,
        "POGOBOT_DIAMETER_CM": 4.975,
        "PIXEL_DIAMETER": 97.01,
        "DEBUG_MODE": True,

        "PLOT_TRAJECTORIES": True,
        "ARENA_XLIM": [0, 2992],
        "ARENA_YLIM": [0, 2976],
        "ARROW_LENGTH_VIS": 100,
        "ARENA_RADIUS": 50,
        "HEAD_WIDTH": 30,
        "HEAD_LENGTH": 30,
        "TRAJECTORY_DPI": 300,
        "PHOTOTAXIS_ANALYSIS": False,
        "PHOTOTAXIS_FAST": True,
        "PHOTOTAXIS_ROI_HALF_SIZE": 70,
        "PHOTOTAXIS_LIGHT_CC_USE_ROIS": True,
        "PHOTOTAXIS_LIGHT_CC_ROI_HALF_SIZE": 110,
        "PHOTOTAXIS_FULL_DETECT_EVERY": 0,
        "DIVIDING_LINE": {
            "X1": 0,
            "Y1": 1360,
            "X2": 2975,
            "Y2": 1507,
            "DARK_REF_X": 1400,
            "DARK_REF_Y": 2200,
        },
        "PHOTOTAXIS_THRESHOLDS": {
            "DARK_THRESHOLD": 100,
            "LIGHT_THRESHOLD": 70,
        },
        "LIGHT_CONNECTED_COMPONENTS": {
            "ENABLED": True,
            "MIN_AREA": 45,
            "CONNECTIVITY": 8,
        },
        "HOUGH_CIRCLES": {
            "ENABLED": True,
            "DP": 1.0,
            "MIN_DIST": 70.0,
            "PARAM1": 10.0,
            "PARAM2": 14.0,
            "MIN_RADIUS": 40,
            "MAX_RADIUS": 60,
            "FALLBACK_PARAM2": 12.0,
            "FALLBACK_MIN_DIST": 65.0,
            "FALLBACK_MIN_RADIUS": 38,
            "FALLBACK_MAX_RADIUS": 62,
        },
        "RGB_ID_ANALYSIS": False,
        "RGB_ID": {
            "ROI_SIZE": 30,
            "GAMMA": 80,
            "RHO": 20,
            "DETECT_ON_FRAME": 0,
        },
        
    }
    
    _background_cache = {}  # Cache backgrounds by path
    _config_cache = {}      # Cache configs by path

    def __init__(self, video_path, background_path, save_path, config_path, frame_visualize = None):

        """
        Initialize class by defining the attributes variables.

        Parameters
        ----------
            video_path (str):
                path to the input video to analyze (.mp4).
            background_path (str):
                path to the background image (.bmp).
            save_path (str):
                path where the output .csv will be saved.
            config_path (str):
                path to .yaml configuration file.
            frame_visualize (int | list[int] | None):
                frames for which to visualize contours
                (optional = None).
        """

        self.video_path = video_path
        self.background_path = background_path
        self.save_path = save_path
        self.config_path = config_path

        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Load or retrieve cached config
        if config_path not in self._config_cache:
            self._config_cache[config_path] = self._load_config(yaml_config)
        self.config = self._config_cache[config_path]

        # Normalize frames-to-visualize: merge CLI (if any) with .yaml list
        cli_frames = frame_visualize
        yaml_frames = self.config.get("VISUALIZE_CONTOURS_FRAMES", [])
        self.frames_to_visualize = set()

        def _add_frames(fr):
            if fr is None:
                return
            if isinstance(fr, int):
                self.frames_to_visualize.add(fr)
            else:
                try:
                    for k in fr:
                        self.frames_to_visualize.add(int(k))
                except TypeError:
                    pass

        _add_frames(yaml_frames)
        _add_frames(cli_frames)

        self.df = pd.DataFrame(columns=["frame", "x", "y", "theta"])
        self._data_rows = []
        self._record_start_frame = 0

        self._phototaxis_cache = None
        self._prev_phototaxis_centers = None
        self._timings = defaultdict(float)
        self._timings_count = 0

        self.DIVIDING_LINE["REGION_TEST_ON_UNFLIPPED_COORDS"] = bool(
        self.RGB_ID.get("REGION_TEST_ON_UNFLIPPED_COORDS", True)
)

    def _load_config(self, yaml_config: dict):

        """
        Merge default processing parameters with .yaml configuration.
        .yaml has the priority.

        Parameters
        ----------
            yaml_config : dict
                dictionary of configuration parameters loaded from .yaml.
        """

        yaml_config = yaml_config or {}

        def coerce_bool(v):
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "yes", "1", "on"):  return True
                if s in ("false", "no", "0", "off"): return False
            return v

        # Normalize keys and coerce booleans recursively if needed
        normalized_yaml = {}
        for k, v in yaml_config.items():
            key_up = str(k).upper()
            if isinstance(v, dict):
                normalized_yaml[key_up] = {str(kk).upper(): coerce_bool(vv) for kk, vv in v.items()}
            else:
                normalized_yaml[key_up] = coerce_bool(v)

        merged = {**self.DEFAULTS, **normalized_yaml}
        # Keep plotting options grouped in YAML while exposing them at the
        # top level to the legacy visualization helpers.
        plotting_config = normalized_yaml.get("PLOTTING")
        if isinstance(plotting_config, dict):
            merged.update(plotting_config)
        for key, value in merged.items():
            setattr(self, key, value)
        return merged

    def _load_video_and_background(self):

        """
        Load video and background image (all the channels, not only the blue)
        """

        self.video = cv2.VideoCapture(self.video_path)
        
        # use cache
        if self.background_path not in self._background_cache:
            self._background_cache[self.background_path] = cv2.imread(self.background_path)
            self._background_cache[self.background_path] = cv2.flip(self._background_cache[self.background_path], 0)
        self.background = self._background_cache[self.background_path]

    def _create_mask(self, rectangular=False):

        """
        Create a circular or rectangular binary mask for the arena.

        Return
        ------
        mask (np.ndarray):
            Binary mask with arena pixels set to 255 and
            outside pixels set to 0.
        """

        mask = np.zeros(self.background.shape[:2], dtype=np.uint8)
        if rectangular:
            cx, cy = self.CENTER 
            corner_topleft  = (int(cx - self.WIDTH  / 2), int(cy - self.HEIGHT / 2))
            corner_botright = (int(cx + self.WIDTH  / 2), int(cy + self.HEIGHT / 2))
            cv2.rectangle(mask, corner_topleft, corner_botright, 255, -1)
        else:
            cv2.circle(mask, tuple(self.CENTER), int(self.RADIUS), 255, -1)
        return mask

    def skip_frame(self, n: int):

        """
        Skip a frame during processing and insert
        placeholder data.

        This method records NaNs for positions and
        orientations when detection fails, ensuring
        that the output dataframe preserves frame continuity
        with N_POGO entries per frame.

        Parameters
        ----------
        n (int):
            index of frame to skip

        Return
        ------
        n (int):
            index of next frame to analyze
        """
        
        x = [np.nan] * self.N_POGO
        y = [np.nan] * self.N_POGO
        thetas = [np.nan] * self.N_POGO

        self._append_data_rows(n, x, y, thetas)
        return n + 1

    def _append_data_rows(self, frame, x, y, thetas):
        """Append one frame's detections without repeatedly copying a DataFrame."""
        if frame < self._record_start_frame:
            return
        if len(x) == 0 or len(y) == 0 or len(thetas) == 0:
            return

        self._data_rows.extend(
            (frame, x_value, y_value, theta_value)
            for x_value, y_value, theta_value in zip(x, y, thetas)
        )
    
    def _adjust_detection(
        self,
        diff=None,
        *,
        mode="classic",
        frame=None,
        mask=None,
        phot=None,
        current_circles=None,
    ):
        """
        Dispatch detection recovery to the classic or phototaxis branch.
        """
        if mode == "classic":
            if diff is None:
                raise ValueError("Classic adjustment requires `diff`.")
            return self._adjust_detection_classic(diff)

        if mode == "phototaxis":
            if frame is None or mask is None:
                raise ValueError("Phototaxis adjustment requires `frame` and `mask`.")
            return self._adjust_detection_phototaxis(
                frame,
                mask,
                phot=phot,
                current_circles=current_circles,
            )

        raise ValueError(f"Unsupported detection mode: {mode}")


    def _adjust_detection_classic(self, diff):
        """
        Attempt to recover correct Pogobot detections for the classic pipeline.

        Returns
        -------
        result : dict
            Keys: contours, circles, x, y, thetas, thresh, diff, dark_bin,
            light_bin, success
        """
        base_thresh = self.THRESHOLD
        max_offset = self.MAX_OFFSET

        offsets = list(range(1, max_offset + 1))
        candidates = [base_thresh + o for o in offsets] + [base_thresh - o for o in offsets]

        last_result = {
            "contours": [],
            "circles": np.empty((0, 3), dtype=np.int32),
            "x": [],
            "y": [],
            "thetas": [],
            "thresh": None,
            "diff": diff,
            "dark_bin": None,
            "light_bin": None,
            "success": False,
        }

        for t in candidates:
            if t <= 0:
                continue

            thresh = binarize(diff, threshold=t)
            contours = find_contours(thresh, area_params=self.AREAS, peri_params=self.PERIMETERS)
            x, y = get_position(contours)
            thetas = get_all_angles(thresh, y, x)

            result = {
                "contours": contours,
                "circles": np.empty((0, 3), dtype=np.int32),
                "x": x,
                "y": y,
                "thetas": thetas,
                "thresh": thresh,
                "diff": diff,
                "dark_bin": None,
                "light_bin": None,
                "success": len(thetas) == self.N_POGO,
            }

            if result["success"]:
                _progress_log("Success with threshold = " + str(t) + ".")
                return result

            last_result = result

        thresh = binarize(diff, threshold=base_thresh)
        contours = find_contours(thresh, area_params=self.FALLBACK_AREA, peri_params=self.FALLBACK_PERIMETERS)
        x, y = get_position(contours)
        thetas = get_all_angles(thresh, y, x)

        result = {
            "contours": contours,
            "circles": np.empty((0, 3), dtype=np.int32),
            "x": x,
            "y": y,
            "thetas": thetas,
            "thresh": thresh,
            "diff": diff,
            "dark_bin": None,
            "light_bin": None,
            "success": len(thetas) == self.N_POGO,
        }

        if result["success"]:
            _progress_log("Success with wider area/perimeter ranges.")
            return result

        return result if result["thresh"] is not None else last_result

    def _adjust_detection_phototaxis(self, frame, mask, phot=None, current_circles=None):
        """
        Attempt to recover correct Pogobot detections for the phototaxis pipeline.

        Strategy
        --------
        1. Sweep light threshold around the configured value.
        2. Sweep dark threshold around the configured value.
        3. Sweep both together.
        4. For each candidate, run Hough detection with built-in fallback.

        Returns
        -------
        result : dict
            Keys: contours, circles, x, y, thetas, thresh, diff, dark_bin,
            light_bin, success
        """
        base_dark = int(self.PHOTOTAXIS_THRESHOLDS["DARK_THRESHOLD"])
        base_light = int(self.PHOTOTAXIS_THRESHOLDS["LIGHT_THRESHOLD"])
        max_offset = min(int(self.MAX_OFFSET), 10)

        candidate_pairs = []

        def _add_pair(dt, lt):
            if dt <= 0 or lt <= 0:
                return
            pair = (int(dt), int(lt))
            if pair not in candidate_pairs:
                candidate_pairs.append(pair)

        _add_pair(base_dark, base_light)

        for o in range(1, max_offset + 1):
            _add_pair(base_dark, base_light + o)
            _add_pair(base_dark, base_light - o)
            _add_pair(base_dark + o, base_light)
            _add_pair(base_dark - o, base_light)
            _add_pair(base_dark + o, base_light + o)
            _add_pair(base_dark - o, base_light - o)

        last_result = {
            "contours": [],
            "circles": np.empty((0, 3), dtype=np.int32),
            "x": [],
            "y": [],
            "thetas": [],
            "thresh": None,
            "diff": None,
            "dark_bin": None,
            "light_bin": None,
            "success": False,
        }

        if phot is None:
            phot = self._process_phototaxis_frame(frame, mask)

        if self._phototaxis_cache is None:
            self._prepare_phototaxis_static_data(mask)

        dark_diff = phot.get("dark_diff")
        light_diff = phot.get("light_diff")
        predicted_centers = self._get_phototaxis_predicted_centers()
        missing_centers = self._get_missing_predicted_centers(
            current_circles,
            predicted_centers,
        )

        # First retry locally around the last valid robot positions. This keeps
        # the exact-N requirement while avoiding a full-frame Hough sweep for
        # ordinary one- or two-robot misses. Only missing centers are retried.
        if missing_centers and dark_diff is not None and light_diff is not None:
            for dark_t, light_t in candidate_pairs:
                candidate_phot = self._build_phototaxis_frame_from_diffs(
                    dark_diff,
                    light_diff,
                    dark_t,
                    light_t,
                    predicted_centers=missing_centers,
                )
                circles, x, y, thetas = self._detect_phototaxis_bots_roi(
                    candidate_phot["out"],
                    allow_full_fallback=False,
                    predicted_centers=missing_centers,
                )
                base_circles = (
                    current_circles
                    if current_circles is not None
                    else np.empty((0, 3), dtype=np.int32)
                )
                merged_circles = np.vstack([base_circles, circles]) if len(circles) else base_circles
                merged_circles = self._match_circles_to_predicted_centers(
                    merged_circles,
                    predicted_centers,
                    int(self.PHOTOTAXIS_ROI_HALF_SIZE),
                )
                x, y = circles_to_centers(merged_circles)
                thetas = get_all_angles(candidate_phot["out"], y, x)
                result = {
                    "contours": [],
                    "circles": merged_circles,
                    "x": x,
                    "y": y,
                    "thetas": thetas,
                    "thresh": candidate_phot["out"],
                    "diff": candidate_phot["out"],
                    "dark_bin": candidate_phot["dark_bin"],
                    "light_bin": candidate_phot["light_bin"],
                    "success": len(merged_circles) == self.N_POGO and len(thetas) == self.N_POGO,
                }
                if result["success"]:
                    _progress_log(f"Success with local phototaxis thresholds dark={dark_t}, light={light_t}.")
                    return result
                last_result = result

        # If local recovery cannot assign every expected robot, retain the
        # original global recovery path, but reuse the already computed image
        # differences instead of rebuilding masks/background differences for
        # every threshold pair.
        for dark_t, light_t in candidate_pairs:
            if dark_diff is not None and light_diff is not None:
                candidate_phot = self._build_phototaxis_frame_from_diffs(
                    dark_diff,
                    light_diff,
                    dark_t,
                    light_t,
                    predicted_centers=None,
                )
            else:
                candidate_phot = process_phototaxis_frame(
                    frame=frame,
                    background=self.background,
                    arena_mask=mask,
                    line_cfg=self.DIVIDING_LINE,
                    dark_threshold=dark_t,
                    light_threshold=light_t,
                    light_cc_enabled=bool(self.LIGHT_CONNECTED_COMPONENTS.get("ENABLED", True)),
                    light_min_area=int(self.LIGHT_CONNECTED_COMPONENTS.get("MIN_AREA", 45)),
                    light_connectivity=int(self.LIGHT_CONNECTED_COMPONENTS.get("CONNECTIVITY", 8)),
                    debug=bool(self.config.get("DEBUG_MODE", False)),
                )

            circles = detect_circles_with_fallback(
                candidate_phot["out"], self.N_POGO, self.HOUGH_CIRCLES
            )
            x, y = circles_to_centers(circles)
            thetas = get_all_angles(candidate_phot["out"], y, x)

            result = {
                "contours": [],
                "circles": circles,
                "x": x,
                "y": y,
                "thetas": thetas,
                "thresh": candidate_phot["out"],
                "diff": candidate_phot["out"],
                "dark_bin": candidate_phot["dark_bin"],
                "light_bin": candidate_phot["light_bin"],
                "success": len(circles) == self.N_POGO and len(thetas) == self.N_POGO,
            }

            if result["success"]:
                _progress_log(f"Success with phototaxis thresholds dark={dark_t}, light={light_t}.")
                return result

            last_result = result

        return last_result

    def _process_phototaxis_frame(self, frame, mask):
        if self.PHOTOTAXIS_FAST:
            if self._phototaxis_cache is None:
                self._prepare_phototaxis_static_data(mask)
            return self._process_phototaxis_frame_fast(frame)

        return process_phototaxis_frame(
            frame=frame,
            background=self.background,
            arena_mask=mask,
            line_cfg=self.DIVIDING_LINE,
            dark_threshold=self.PHOTOTAXIS_THRESHOLDS['DARK_THRESHOLD'],
            light_threshold=self.PHOTOTAXIS_THRESHOLDS['LIGHT_THRESHOLD'],
            light_cc_enabled=bool(self.LIGHT_CONNECTED_COMPONENTS.get('ENABLED', True)),
            light_min_area=int(self.LIGHT_CONNECTED_COMPONENTS.get('MIN_AREA', 45)),
            light_connectivity=int(self.LIGHT_CONNECTED_COMPONENTS.get('CONNECTIVITY', 8)),
            debug=bool(self.config.get('DEBUG_MODE', False)),
        )

    def _detect_phototaxis_bots(self, binary, frame_idx=None):
        full_every = int(self.PHOTOTAXIS_FULL_DETECT_EVERY)
        force_full = full_every > 0 and frame_idx is not None and frame_idx % full_every == 0

        if self.PHOTOTAXIS_FAST and not force_full:
            t_roi = self._tic()
            circles, x, y, thetas = self._detect_phototaxis_bots_roi(binary)
            self._add_timing("phot_detect_roi", t_roi)
        else:
            t_full = self._tic()
            circles = detect_circles_with_fallback(binary, self.N_POGO, self.HOUGH_CIRCLES)
            x, y = circles_to_centers(circles)
            self._add_timing("phot_detect_full_hough", t_full)

            t_theta = self._tic()
            thetas = get_all_angles(binary, y, x)
            self._add_timing("phot_theta", t_theta)

        self._update_phototaxis_state(x, y)
        return circles, x, y, thetas

    def _prepare_phototaxis_static_data(self, arena_mask):
        if self._phototaxis_cache is not None:
            return self._phototaxis_cache

        line_cfg = self.DIVIDING_LINE
        dark_side_mask, light_side_mask = create_line_region_masks(
            self.background.shape,
            line_cfg["X1"], line_cfg["Y1"], line_cfg["X2"], line_cfg["Y2"],
            line_cfg["DARK_REF_X"], line_cfg["DARK_REF_Y"],
        )

        dark_mask = cv2.bitwise_and(arena_mask, dark_side_mask)
        light_mask = cv2.bitwise_and(arena_mask, light_side_mask)

        background_blue = to_blue_channel(self.background)
        dark_background_masked = cv2.bitwise_and(background_blue, background_blue, mask=dark_mask)
        light_background_masked = cv2.bitwise_and(self.background, self.background, mask=light_mask)

        self._phototaxis_cache = {
            "arena_mask": arena_mask,
            "dark_side_mask": dark_side_mask,
            "light_side_mask": light_side_mask,
            "dark_mask": dark_mask,
            "light_mask": light_mask,
            "background_blue": background_blue,
            "dark_background_masked": dark_background_masked,
            "light_background_masked": light_background_masked,
        }
        return self._phototaxis_cache

    def _get_phototaxis_predicted_centers(self):
        if self._prev_phototaxis_centers is not None and len(self._prev_phototaxis_centers) > 0:
            return self._prev_phototaxis_centers
        return None
    
    def _process_phototaxis_frame_fast(self, frame):
        cache = self._phototaxis_cache

        t = self._tic()
        dark_frame = to_blue_channel(frame)
        self._add_timing("phot_dark_blue", t)

        t = self._tic()
        dark_frame_masked = cv2.bitwise_and(dark_frame, dark_frame, mask=cache["dark_mask"])
        dark_diff = get_difference(
            dark_frame_masked,
            cache["dark_background_masked"],
            bool(self.config.get("DEBUG_MODE", False)),
        )
        dark_bin = binarize(
            dark_diff,
            threshold=self.PHOTOTAXIS_THRESHOLDS["DARK_THRESHOLD"],
        )
        self._add_timing("phot_dark_branch", t)

        t = self._tic()
        light_frame_masked = cv2.bitwise_and(frame, frame, mask=cache["light_mask"])
        light_diff = get_difference(
            light_frame_masked,
            cache["light_background_masked"],
            bool(self.config.get("DEBUG_MODE", False)),
        )
        light_bin = binarize(
            light_diff,
            threshold=self.PHOTOTAXIS_THRESHOLDS["LIGHT_THRESHOLD"],
        )
        self._add_timing("phot_light_branch", t)

        if bool(self.LIGHT_CONNECTED_COMPONENTS.get("ENABLED", True)):
            t = self._tic()
            predicted_centers = self._get_phototaxis_predicted_centers()

            if bool(self.PHOTOTAXIS_LIGHT_CC_USE_ROIS) and predicted_centers is not None:
                light_bin = filter_small_components_in_rois(
                    light_bin,
                    centers=predicted_centers,
                    half_size=int(self.PHOTOTAXIS_LIGHT_CC_ROI_HALF_SIZE),
                    min_area=int(self.LIGHT_CONNECTED_COMPONENTS.get("MIN_AREA", 45)),
                    connectivity=int(self.LIGHT_CONNECTED_COMPONENTS.get("CONNECTIVITY", 8)),
                )
            else:
                light_bin = filter_small_components(
                    light_bin,
                    int(self.LIGHT_CONNECTED_COMPONENTS.get("MIN_AREA", 45)),
                    int(self.LIGHT_CONNECTED_COMPONENTS.get("CONNECTIVITY", 8)),
                )

            self._add_timing("phot_light_cc", t)

        t = self._tic()
        out = cv2.bitwise_or(dark_bin, light_bin)
        out[cache["arena_mask"] == 0] = 0
        self._add_timing("phot_recombine", t)

        return {
            "dark_bin": dark_bin,
            "light_bin": light_bin,
            "out": out,
            "dark_diff": dark_diff,
            "light_diff": light_diff,
        }

    def _build_phototaxis_frame_from_diffs(
        self,
        dark_diff,
        light_diff,
        dark_threshold,
        light_threshold,
        predicted_centers=None,
    ):
        """Rebuild thresholded phototaxis images from cached differences."""
        cache = self._phototaxis_cache
        dark_bin = binarize(dark_diff, threshold=dark_threshold)
        light_bin = binarize(light_diff, threshold=light_threshold)

        if bool(self.LIGHT_CONNECTED_COMPONENTS.get("ENABLED", True)):
            min_area = int(self.LIGHT_CONNECTED_COMPONENTS.get("MIN_AREA", 45))
            connectivity = int(self.LIGHT_CONNECTED_COMPONENTS.get("CONNECTIVITY", 8))
            if bool(self.PHOTOTAXIS_LIGHT_CC_USE_ROIS) and predicted_centers is not None:
                light_bin = filter_small_components_in_rois(
                    light_bin,
                    centers=predicted_centers,
                    half_size=int(self.PHOTOTAXIS_LIGHT_CC_ROI_HALF_SIZE),
                    min_area=min_area,
                    connectivity=connectivity,
                )
            else:
                light_bin = filter_small_components(light_bin, min_area, connectivity)

        out = cv2.bitwise_or(dark_bin, light_bin)
        out[cache["arena_mask"] == 0] = 0
        return {
            "dark_bin": dark_bin,
            "light_bin": light_bin,
            "out": out,
            "dark_diff": dark_diff,
            "light_diff": light_diff,
        }

    def _clip_roi(self, cx, cy, half_size, shape):
        h, w = shape[:2]
        x1 = max(0, int(cx - half_size))
        y1 = max(0, int(cy - half_size))
        x2 = min(w, int(cx + half_size + 1))
        y2 = min(h, int(cy + half_size + 1))
        return x1, y1, x2, y2

    def _detect_phototaxis_bots_roi(
        self,
        binary,
        allow_full_fallback=True,
        predicted_centers=None,
    ):
        centers = predicted_centers
        if centers is None:
            centers = self._prev_phototaxis_centers

        if not centers:
            circles = detect_circles_with_fallback(binary, self.N_POGO, self.HOUGH_CIRCLES)
            x, y = circles_to_centers(circles)
            thetas = get_all_angles(binary, y, x)
            return circles, x, y, thetas

        half_size = int(self.PHOTOTAXIS_ROI_HALF_SIZE)
        circles_all = []

        t_loop = self._tic()
        for cx, cy in centers:
            x1, y1, x2, y2 = self._clip_roi(cx, cy, half_size, binary.shape)
            roi = binary[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            circles_roi = detect_hough_circles(
                roi,
                dp=self.HOUGH_CIRCLES["DP"],
                min_dist=max(20.0, self.HOUGH_CIRCLES["MIN_DIST"] * 0.4),
                param1=self.HOUGH_CIRCLES["PARAM1"],
                param2=self.HOUGH_CIRCLES["PARAM2"],
                min_radius=self.HOUGH_CIRCLES["MIN_RADIUS"],
                max_radius=self.HOUGH_CIRCLES["MAX_RADIUS"],
            )


            if len(circles_roi) == 0:
                circles_roi = detect_hough_circles(
                    roi,
                    dp=self.HOUGH_CIRCLES["DP"],
                    min_dist=max(15.0, self.HOUGH_CIRCLES.get("FALLBACK_MIN_DIST", self.HOUGH_CIRCLES["MIN_DIST"]) * 0.4),
                    param1=self.HOUGH_CIRCLES["PARAM1"],
                    param2=self.HOUGH_CIRCLES.get("FALLBACK_PARAM2", self.HOUGH_CIRCLES["PARAM2"]),
                    min_radius=self.HOUGH_CIRCLES.get("FALLBACK_MIN_RADIUS", self.HOUGH_CIRCLES["MIN_RADIUS"]),
                    max_radius=self.HOUGH_CIRCLES.get("FALLBACK_MAX_RADIUS", self.HOUGH_CIRCLES["MAX_RADIUS"]),
                )

            if len(circles_roi) == 0:
                continue

            circles_roi = circles_roi.copy()
            circles_roi[:, 0] += x1
            circles_roi[:, 1] += y1
            circles_all.extend(circles_roi)

        self._add_timing("phot_roi_loop", t_loop)
        circles = self._match_circles_to_predicted_centers(
            circles_all,
            centers,
            half_size,
        )
        if len(circles) < len(centers) and allow_full_fallback:
            circles = detect_circles_with_fallback(binary, self.N_POGO, self.HOUGH_CIRCLES)

        x, y = circles_to_centers(circles)
        t_theta = self._tic()
        thetas = get_all_angles(binary, y, x)
        self._add_timing("phot_theta", t_theta)
        return circles, x, y, thetas

    def _get_missing_predicted_centers(self, circles, predicted_centers):
        """Return expected centers that were not represented by current circles."""
        if predicted_centers is None:
            return []
        if circles is None or len(circles) == 0:
            return list(predicted_centers)

        unique_circles = deduplicate_circles(np.asarray(circles, dtype=np.float32))
        if len(unique_circles) == 0:
            return list(predicted_centers)

        targets = np.asarray(predicted_centers, dtype=np.float32)
        candidates = unique_circles[:, :2].astype(np.float32)
        distances = np.sqrt(np.sum((targets[:, None, :] - candidates[None, :, :]) ** 2, axis=2))
        used_targets = set()
        used_candidates = set()
        max_distance = int(self.PHOTOTAXIS_ROI_HALF_SIZE)

        for distance, target_idx, candidate_idx in sorted(
            (float(distances[target_idx, candidate_idx]), target_idx, candidate_idx)
            for target_idx in range(len(targets))
            for candidate_idx in range(len(candidates))
        ):
            if distance > max_distance:
                break
            if target_idx in used_targets or candidate_idx in used_candidates:
                continue
            used_targets.add(target_idx)
            used_candidates.add(candidate_idx)

        return [center for idx, center in enumerate(predicted_centers) if idx not in used_targets]

    def _match_circles_to_predicted_centers(self, circles, predicted_centers, max_distance):
        """Assign at most one circle to each expected robot center."""
        if predicted_centers is None or len(predicted_centers) == 0 or len(circles) == 0:
            return np.empty((0, 3), dtype=np.int32)

        unique_circles = deduplicate_circles(np.asarray(circles, dtype=np.float32))
        if len(unique_circles) == 0:
            return np.empty((0, 3), dtype=np.int32)

        targets = np.asarray(predicted_centers, dtype=np.float32)
        candidates = unique_circles[:, :2].astype(np.float32)
        distances = np.sqrt(np.sum((targets[:, None, :] - candidates[None, :, :]) ** 2, axis=2))

        assignments = []
        used_targets = set()
        used_candidates = set()
        for distance, target_idx, candidate_idx in sorted(
            (float(distances[target_idx, candidate_idx]), target_idx, candidate_idx)
            for target_idx in range(len(targets))
            for candidate_idx in range(len(candidates))
        ):
            if distance > max_distance:
                break
            if target_idx in used_targets or candidate_idx in used_candidates:
                continue
            used_targets.add(target_idx)
            used_candidates.add(candidate_idx)
            assignments.append((target_idx, candidate_idx))

        assignments.sort()
        if not assignments:
            return np.empty((0, 3), dtype=np.int32)
        return np.asarray(
            [unique_circles[candidate_idx] for _, candidate_idx in assignments],
            dtype=np.int32,
        )

    def _update_phototaxis_state(self, x, y):
        if len(x) == self.N_POGO and len(y) == self.N_POGO:
            self._prev_phototaxis_centers = list(zip(x, y))

    def _tic(self):
        return time.perf_counter()

    def _add_timing(self, key, t0):
        self._timings[key] += (time.perf_counter() - t0)

    def _print_timing_summary(self, every=5000):
        if self._timings_count == 0 or self._timings_count % every != 0:
            return

        total = sum(self._timings.values())
        _progress_log(f"\n--- Timing summary after {self._timings_count} frames ---")
        for key, value in sorted(self._timings.items(), key=lambda kv: kv[1], reverse=True):
            avg_ms = 1000.0 * value / self._timings_count
            frac = (100.0 * value / total) if total > 0 else 0.0
            _progress_log(f"{key:28s}: {avg_ms:8.3f} ms/frame   ({frac:5.1f}%)")
        _progress_log("---------------------------------------------\n")



    def _detect_static_rgb_bots(self, frame_raw, mask, background_masked, gamma, rho, frame_idx):
        """
        Detect robots once and compute LED positions.

        Returns
        -------
        x, y, thetas, led_positions : np.ndarray
        """
        if self.PHOTOTAXIS_ANALYSIS:
            phot = self._process_phototaxis_frame(frame_raw, mask)
            thresh = phot["out"]

            circles, x, y, thetas = self._detect_phototaxis_bots(thresh, frame_idx=frame_idx)

            ok = (
                len(circles) == self.N_POGO
                and len(x) == self.N_POGO
                and len(y) == self.N_POGO
                and len(thetas) == self.N_POGO
            )
        else:
            frame_blue = to_blue_channel(frame_raw)
            frame_masked = cv2.bitwise_and(frame_blue, frame_blue, mask=mask)

            diff = get_difference(
                frame_masked,
                background_masked,
                False,
            )
            thresh = binarize(diff, threshold=self.THRESHOLD)
            contours = find_contours(
                thresh,
                area_params=self.AREAS,
                peri_params=self.PERIMETERS,
            )
            x, y = get_position(contours)
            thetas = get_all_angles(thresh, y, x)

            ok = x is not None and len(x) == self.N_POGO and len(thetas) == self.N_POGO

            if not ok:
                result = self._adjust_detection(diff=diff, mode="classic")
                x = result["x"]
                y = result["y"]
                thetas = result["thetas"]

                ok = x is not None and len(x) == self.N_POGO and len(thetas) == self.N_POGO

        if not ok:
            raise RuntimeError(
                f"Frame {frame_idx}: could not detect all {self.N_POGO} robots for RGB-ID processing."
            )

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        thetas = np.asarray(thetas, dtype=float)
        led_positions = np.asarray(
            compute_led_positions(x, y, thetas, gamma=gamma, rho=rho),
            dtype=float,
        )

        return x, y, thetas, led_positions


    def process_rgb_id(self):
        """
        Process a static RGB-ID experiment.

        Robots are detected once on `DETECT_ON_FRAME`. Their pose and LED
        positions are then reused for all frames to measure RGB values and save
        a single CSV with columns:
        `frame, time, id, x, y, theta, led_x, led_y, R, G, B`.
        """
        start = time.time()

        self._load_video_and_background()
        mask = self._create_mask(rectangular=self.RECT_MASK)
        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        rgb_cfg = self.config.get("RGB_ID", {})
        roi_size = int(rgb_cfg.get("ROI_SIZE", 30))
        gamma = float(rgb_cfg.get("GAMMA", 80))
        rho = float(rgb_cfg.get("RHO", 20))
        detect_on_frame = int(rgb_cfg.get("DETECT_ON_FRAME", 0))

        debug_enabled = bool(rgb_cfg.get("DEBUG_RGB", False))
        debug_frames = set(rgb_cfg.get("DEBUG_RGB_FRAMES", []))
        debug_every_n = int(rgb_cfg.get("DEBUG_RGB_EVERY_N", 1))
        debug_dir = Path(self.save_path).with_name(str(rgb_cfg.get("DEBUG_RGB_DIR", "rgb_id_debug")))
        debug_zoom_size = int(rgb_cfg.get("DEBUG_RGB_ZOOM_SIZE", 96))
        debug_contact_cols = int(rgb_cfg.get("DEBUG_RGB_CONTACT_COLS", 6))
        debug_gif_name = str(rgb_cfg.get("DEBUG_RGB_GIF", "leds.gif"))
        debug_gif_duration = float(rgb_cfg.get("DEBUG_RGB_GIF_DURATION", 0.12))

        fps = float(self.config["FPS"])
        pogobot_diameter_cm = float(self.config["POGOBOT_DIAMETER_CM"])
        pixel_diameter = float(self.config["PIXEL_DIAMETER"])

        rgb_path = Path(self.save_path).with_name("RGB_ID.csv")
        rgb_rows = []

        background_blue = to_blue_channel(self.background)
        background_masked = cv2.bitwise_and(background_blue, background_blue, mask=mask)

        x = y = thetas = led_positions = ids = None
        led_gif_frames = []
        n = 0

        try:
            with tqdm(total=total_frames, desc="Processing RGB-ID video", unit="frame") as pbar:
                while self.video.isOpened():
                    ret, frame = self.video.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 0)
                    frame_raw = frame.copy()

                    if n == detect_on_frame:
                        x, y, thetas, led_positions = self._detect_static_rgb_bots(
                            frame_raw=frame_raw,
                            mask=mask,
                            background_masked=background_masked,
                            gamma=gamma,
                            rho=rho,
                            frame_idx=n,
                        )
                        ids = np.arange(len(x), dtype=int)

                    if led_positions is not None:
                        frame_csv_rows, frame_debug_rows = _measure_frame_rows(
                            frame=frame_raw,
                            frame_idx=n,
                            ids=ids,
                            x=x,
                            y=y,
                            thetas=thetas,
                            led_positions=led_positions,
                            roi_size=roi_size,
                            fps=fps,
                            pogobot_diameter_cm=pogobot_diameter_cm,
                            pixel_diameter=pixel_diameter,
                            line_cfg=getattr(self, "DIVIDING_LINE", None) if debug_enabled else None,
                            coords_are_flipped=True,
                        )

                        rgb_rows.extend(frame_csv_rows)

                        should_debug = (
                            debug_enabled and (
                                (not debug_frames and n % max(1, debug_every_n) == 0) or
                                (n in debug_frames)
                            )
                        )

                        if should_debug:
                            arena_path = debug_dir / "arena" / f"frame_{n:05d}.png"
                            leds_path = debug_dir / "leds" / f"frame_{n:05d}.png"

                            save_rgb_arena_debug_view(
                                frame=frame_raw,
                                rows_for_frame=frame_debug_rows,
                                out_path=arena_path,
                                roi_size=roi_size,
                                line_cfg=getattr(self, "DIVIDING_LINE", None),
                            )

                            sheet = save_led_contact_sheet(
                                frame=frame_raw,
                                rows_for_frame=frame_debug_rows,
                                out_path=leds_path,
                                roi_size=roi_size,
                                zoom_size=debug_zoom_size,
                                cols=debug_contact_cols,
                            )
                            if sheet is not None:
                                led_gif_frames.append(sheet)

                    n += 1
                    pbar.update(1)
        finally:
            self.video.release()

        if not rgb_rows:
            raise RuntimeError("No RGB measurements were collected.")

        df_rgb = pd.DataFrame(
            rgb_rows,
            columns=["frame", "time", "id", "x", "y", "theta", "led_x", "led_y", "R", "G", "B"],
        )
        df_rgb.to_csv(rgb_path, index=False)

        if debug_enabled and led_gif_frames:
            save_led_debug_gif(
                frames_bgr=led_gif_frames,
                out_path=debug_dir / debug_gif_name,
                duration=debug_gif_duration,
            )

        elapsed = time.time() - start
        _progress_log(f"RGB CSV: {rgb_path}")
        if debug_enabled:
            _progress_log(f"Debug directory: {debug_dir}")
        _progress_log(f"Processed {n} frames in {elapsed:.2f}s")

        return df_rgb

    def _finalize_dataframe(self, df):
        """Run global linking, unit conversion, and final output writing."""
        self.df = df
        df_tracked = track_objects(
            self.df,
            search_range=self.SEARCH_RANGE,
            memory=self.MEMORY,
        )

        df_transformed = convert_datas(
            df_tracked,
            self.config["FPS"],
            self.config["POGOBOT_DIAMETER_CM"],
            self.config["PIXEL_DIAMETER"],
        )
        df_transformed.to_csv(self.save_path, index=False)

        if bool(self.config.get("PLOT_TRAJECTORIES", False)):
            plot_trajectories(
                self.save_path,
                "Trajectories",
                self.config,
                bg_path=self.background_path,
            )

        return df_transformed



    def process(
        self,
        frame_start=0,
        frame_end=None,
        warmup_start=None,
        finalize=True,
        show_progress=True,
        progress_position=0,
        progress_desc=None,
    ):
        """
        Run the video processing pipeline, optionally over a frame range.

        `warmup_start` allows chunk workers to process context frames before
        `frame_start` without including those detections in their output.
        """
        start = time.time()
        self._load_video_and_background()
        mask = self._create_mask(rectangular=self.RECT_MASK)

        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_start = int(frame_start)
        frame_end = total_frames if frame_end is None else int(frame_end)
        if not 0 <= frame_start <= frame_end <= total_frames:
            raise ValueError("Invalid frame range.")

        warmup_start = frame_start if warmup_start is None else int(warmup_start)
        warmup_start = max(0, min(warmup_start, frame_start))
        self._record_start_frame = frame_start
        self._data_rows = []
        self._prev_phototaxis_centers = None
        self._phototaxis_cache = None

        self.video.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
        n = warmup_start

        if not self.PHOTOTAXIS_ANALYSIS:
            background_blue = to_blue_channel(self.background)
            background_masked = cv2.bitwise_and(background_blue, background_blue, mask=mask)

        with tqdm(
            total=frame_end - frame_start,
            desc=progress_desc or "Processing video",
            unit="frame",
            disable=not show_progress,
            position=progress_position,
            leave=True,
        ) as pbar:
            while self.video.isOpened() and n < frame_end:

                t_frame = self._tic()
                ret, frame = self.video.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 0)

                contours = []
                circles = np.empty((0, 3), dtype=np.int32)
                dark_bin = None
                light_bin = None

                if self.PHOTOTAXIS_ANALYSIS:
                    t_phot = self._tic()
                    phot = self._process_phototaxis_frame(frame, mask)
                    self._add_timing("phot_preprocess", t_phot)

                    diff = phot["out"]
                    thresh = phot["out"]
                    dark_bin = phot["dark_bin"]
                    light_bin = phot["light_bin"]

                    t_detect = self._tic()
                    circles, x, y, thetas = self._detect_phototaxis_bots(thresh, frame_idx=n)
                    self._add_timing("phot_detect_total", t_detect)

                    detected_ok = (len(circles) == self.N_POGO and len(thetas) == self.N_POGO)
                    #xif n == 0:
                        #debug_frame( #to remove
                            #frame,
                            #diff,
                            #thresh,
                            #title=f"Phototaxis debug frame {n}",
                            #mode="phototaxis",
                            #dark_bin=dark_bin,
                            #light_bin=light_bin,
                            #circles=circles,
                        #)

                else:
                    frame_blue = to_blue_channel(frame)
                    frame_masked = cv2.bitwise_and(frame_blue, frame_blue, mask=mask)
                    diff = get_difference(frame_masked, background_masked, bool(self.config.get("DEBUG_MODE", False)))
                    thresh = binarize(diff, threshold=self.THRESHOLD)
                    contours = find_contours(thresh, area_params=self.AREAS, peri_params=self.PERIMETERS)
                    x, y = get_position(contours)
                    thetas = get_all_angles(thresh, y, x)
                    detected_ok = (len(thetas) == self.N_POGO)

                if not detected_ok:
                    _progress_log(f"Frame {n}: detection mismatch. Attempting fallback...")

                    if self.PHOTOTAXIS_ANALYSIS:
                        t_fb = self._tic()
                        result = self._adjust_detection(
                            frame=frame,
                            mask=mask,
                            mode="phototaxis",
                            phot=phot,
                            current_circles=circles,
                        )
                        self._add_timing("phot_fallback", t_fb)
                    else:
                        result = self._adjust_detection(diff=diff, mode="classic")

                    contours = result["contours"]
                    circles = result["circles"]
                    x = result["x"]
                    y = result["y"]
                    thetas = result["thetas"]
                    thresh = result["thresh"]
                    diff = result["diff"]
                    dark_bin = result["dark_bin"]
                    light_bin = result["light_bin"]
                    success = result["success"]

                    if not success:
                        _progress_log(f"Frame {n}: Still {len(thetas)} bots detected. Skipping frame.")

                        if bool(self.config.get("DEBUG_MODE", False)):
                            if self.PHOTOTAXIS_ANALYSIS:
                                debug_frame(
                                    frame,
                                    diff,
                                    thresh,
                                    title=f"Phototaxis debug frame {n}",
                                    mode="phototaxis",
                                    dark_bin=dark_bin,
                                    light_bin=light_bin,
                                    circles=circles,
                                )
                            else:
                                debug_frame(
                                    frame_masked,
                                    diff,
                                    thresh,
                                    contours,
                                    title=f"Debug frame {n}",
                                    mode="classic",
                                )

                        n = self.skip_frame(n)
                        if n >= frame_start:
                            pbar.update(1)
                        self._add_timing("frame_total", t_frame)
                        self._timings_count += 1
                        self._print_timing_summary(every=50000)                     
                        continue

                    

                if n in self.frames_to_visualize:
                    if self.PHOTOTAXIS_ANALYSIS:
                        debug_frame(
                            frame,
                            diff,
                            thresh,
                            title=f"Phototaxis frame {n}",
                            mode="phototaxis",
                            dark_bin=dark_bin,
                            light_bin=light_bin,
                            circles=circles,
                        )
                    else:
                        visualize_contours(diff, contours, x, y, thetas, self.config)
                
                if self.PHOTOTAXIS_ANALYSIS:
                    self._update_phototaxis_state(x, y)  

                self._append_data_rows(n, x, y, thetas)
                self._add_timing("frame_total", t_frame)
                self._timings_count += 1
                self._print_timing_summary(every=5000)
                n += 1
                if n > frame_start:
                    pbar.update(1)

        self.video.release()

        self.df = pd.DataFrame(
            self._data_rows,
            columns=["frame", "x", "y", "theta"],
        )

        if not finalize:
            return self.df

        self._finalize_dataframe(self.df)

        end = time.time()
        print(f"Processed {n} frames in {round(end - start, 2)}s → {self.save_path}")

    def process_parallel(self, workers=2, warmup_frames=50):
        """Process contiguous video chunks in spawned workers.

        Workers write raw detections to temporary CSV files. The parent then
        concatenates the chunks and performs one global Trackpy linking pass.
        """
        if self.config.get("RGB_ID_ANALYSIS", False):
            raise ValueError("Multiprocessing is only supported for process().")
        if self.frames_to_visualize:
            raise ValueError("Frame visualization is not supported with multiprocessing.")

        workers = int(workers)
        warmup_frames = int(warmup_frames)
        if workers < 2:
            return self.process()
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative.")

        metadata_cap = cv2.VideoCapture(self.video_path)
        total_frames = int(metadata_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata_cap.release()
        if total_frames <= 0:
            raise RuntimeError(f"Could not read frames from {self.video_path!r}.")

        workers = min(workers, total_frames)
        boundaries = np.linspace(0, total_frames, workers + 1, dtype=int)

        with tempfile.TemporaryDirectory(prefix="pogotrack_chunks_") as tmp_dir:
            jobs = []
            for index in range(workers):
                frame_start = int(boundaries[index])
                frame_end = int(boundaries[index + 1])
                chunk_path = str(Path(tmp_dir) / f"chunk_{index:03d}.csv")
                jobs.append(
                    (
                        self.video_path,
                        self.background_path,
                        chunk_path,
                        self.config_path,
                        frame_start,
                        frame_end,
                        max(0, frame_start - warmup_frames),
                        index,
                        workers,
                    )
                )

            context = mp.get_context("spawn")
            tqdm_lock = context.RLock()
            tqdm.set_lock(tqdm_lock)
            with context.Pool(
                processes=workers,
                initializer=tqdm.set_lock,
                initargs=(tqdm_lock,),
            ) as pool:
                chunk_paths = pool.map(_process_video_chunk_worker, jobs)

            chunk_frames = [pd.read_csv(path) for path in chunk_paths]

        merged = pd.concat(chunk_frames, ignore_index=True)
        merged = merged.sort_values("frame", kind="stable").reset_index(drop=True)
        self._finalize_dataframe(merged)
        print(f"Processed {total_frames} frames with {workers} workers → {self.save_path}")
