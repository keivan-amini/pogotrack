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
    save_datas,
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
    extract_square_roi,
    measure_rgb_mean,
    save_rgb_datas,
    decode_digit_from_position_and_rgb,
    transform_pose_columns,
    _build_rgb_rows,
)

from src.plot_helpers import (
    debug_frame,
    visualize_contours,
    plot_trajectories,
    visualize_gif,
    save_rgb_debug_visualization,
    save_region_debug_visualization,
    save_rgb_arena_debug_view,
    save_led_debug_gif,
    _measure_frame_rows,
    save_led_contact_sheet
)



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
        "PHOTOTAXIS_ROI_HALF_SIZE": 90,
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
            "STATIC_BOTS": True,
            "SAVE_RGB_CSV": True,
            "RGB_CSV_PREFIX": "RGB_",
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

        self.df = save_datas(self.df, n, x, y, thetas)
        return n + 1
    
    def _adjust_detection(self, diff=None, *, mode="classic", frame=None, mask=None):
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
            return self._adjust_detection_phototaxis(frame, mask)

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
                print("Success with threshold = " + str(t) + ".")
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
            print("Success with wider area/perimeter ranges.")
            return result

        return result if result["thresh"] is not None else last_result

    def _adjust_detection_phototaxis(self, frame, mask):
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

        for dark_t, light_t in candidate_pairs:
            phot = process_phototaxis_frame(
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

            circles = detect_circles_with_fallback(phot["out"], self.N_POGO, self.HOUGH_CIRCLES)
            x, y = circles_to_centers(circles)
            thetas = get_all_angles(phot["out"], y, x)

            result = {
                "contours": [],
                "circles": circles,
                "x": x,
                "y": y,
                "thetas": thetas,
                "thresh": phot["out"],
                "diff": phot["out"],
                "dark_bin": phot["dark_bin"],
                "light_bin": phot["light_bin"],
                "success": len(circles) == self.N_POGO and len(thetas) == self.N_POGO,
            }

            if result["success"]:
                print(f"Success with phototaxis thresholds dark={dark_t}, light={light_t}.")
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

    def _clip_roi(self, cx, cy, half_size, shape):
        h, w = shape[:2]
        x1 = max(0, int(cx - half_size))
        y1 = max(0, int(cy - half_size))
        x2 = min(w, int(cx + half_size + 1))
        y2 = min(h, int(cy + half_size + 1))
        return x1, y1, x2, y2

    def _detect_phototaxis_bots_roi(self, binary):
        if not self._prev_phototaxis_centers:
            circles = detect_circles_with_fallback(binary, self.N_POGO, self.HOUGH_CIRCLES)
            x, y = circles_to_centers(circles)
            thetas = get_all_angles(binary, y, x)
            return circles, x, y, thetas

        half_size = int(self.PHOTOTAXIS_ROI_HALF_SIZE)
        circles_all = []

        t_loop = self._tic()
        for cx, cy in self._prev_phototaxis_centers:
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

            target = np.array([cx - x1, cy - y1], dtype=np.float32)
            centers_roi = circles_roi[:, :2].astype(np.float32)
            d2 = np.sum((centers_roi - target) ** 2, axis=1)
            best = circles_roi[int(np.argmin(d2))].copy()
            best[0] += x1
            best[1] += y1
            circles_all.append(best)

        self._add_timing("phot_roi_loop", t_loop)
        if len(circles_all) == 0:
            circles = detect_circles_with_fallback(binary, self.N_POGO, self.HOUGH_CIRCLES)
        else:
            circles = deduplicate_circles(np.array(circles_all, dtype=np.int32))
            if len(circles) < self.N_POGO:
                full_circles = detect_circles_with_fallback(binary, self.N_POGO, self.HOUGH_CIRCLES)
                if len(full_circles) > len(circles):
                    circles = full_circles

        x, y = circles_to_centers(circles)
        t_theta = self._tic()
        thetas = get_all_angles(binary, y, x)
        self._add_timing("phot_theta", t_theta)
        return circles, x, y, thetas

    def _update_phototaxis_state(self, x, y):
        if len(x) == self.N_POGO and len(y) == self.N_POGO:
            self._prev_phototaxis_centers = list(zip(x, y))
        else:
            self._prev_phototaxis_centers = None

    def _tic(self):
        return time.perf_counter()

    def _add_timing(self, key, t0):
        self._timings[key] += (time.perf_counter() - t0)

    def _print_timing_summary(self, every=5000):
        if self._timings_count == 0 or self._timings_count % every != 0:
            return

        total = sum(self._timings.values())
        print(f"\n--- Timing summary after {self._timings_count} frames ---")
        for key, value in sorted(self._timings.items(), key=lambda kv: kv[1], reverse=True):
            avg_ms = 1000.0 * value / self._timings_count
            frac = (100.0 * value / total) if total > 0 else 0.0
            print(f"{key:28s}: {avg_ms:8.3f} ms/frame   ({frac:5.1f}%)")
        print("---------------------------------------------\n")



    def _detect_static_bots_for_rgb(self, frame, mask, background_masked):
        frame_blue = to_blue_channel(frame)
        frame_masked = cv2.bitwise_and(frame_blue, frame_blue, mask=mask)

        diff = get_difference(
            frame_masked,
            background_masked,
            bool(self.config.get("DEBUG_MODE", False))
        )
        thresh = binarize(diff, threshold=self.THRESHOLD)
        contours = find_contours(thresh, area_params=self.AREAS, peri_params=self.PERIMETERS)
        x, y = get_position(contours)
        thetas = get_all_angles(thresh, y, x)

        if len(thetas) != self.N_POGO:
            result = self._adjust_detection(diff=diff, mode="classic")
            contours = result["contours"]
            x = result["x"]
            y = result["y"]
            thetas = result["thetas"]
            thresh = result["thresh"]
            diff = result["diff"]

        success = len(thetas) == self.N_POGO
        return {
            "success": success,
            "contours": contours,
            "x": x,
            "y": y,
            "thetas": thetas,
            "diff": diff,
            "thresh": thresh,
            "frame_masked": frame_masked,
        }

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
        print(f"RGB CSV: {rgb_path}")
        if debug_enabled:
            print(f"Debug directory: {debug_dir}")
        print(f"Processed {n} frames in {elapsed:.2f}s")

        return df_rgb



    def process(self):
        """
        Run the full video processing pipeline. WRITE DOCSTRINGS!!
        """
        start = time.time()
        self._load_video_and_background()
        mask = self._create_mask(rectangular=self.RECT_MASK)

        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        n = 0

        ### CHANGE, miglioriamo i parametri per questa scena temporale
        #start_frame = int(20 * 14.5 * 60)
        #end_frame = int(20 * 25 * 60)

        #self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        #n = start_frame
        ### CHANGE

        if not self.PHOTOTAXIS_ANALYSIS:
            background_blue = to_blue_channel(self.background)
            background_masked = cv2.bitwise_and(background_blue, background_blue, mask=mask)

        with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
            while self.video.isOpened():

                ### CHANGE
                #if n > end_frame:
                    #break
                ### CHANGE

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
                    print(f"Frame {n}: detection mismatch. Attempting fallback...")

                    if self.PHOTOTAXIS_ANALYSIS:
                        self._prev_phototaxis_centers = None
                        t_fb = self._tic()
                        result = self._adjust_detection(frame=frame, mask=mask, mode="phototaxis")
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
                        print(f"Frame {n}: Still {len(thetas)} bots detected. Skipping frame.")

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

                self.df = save_datas(self.df, n, x, y, thetas)
                self._add_timing("frame_total", t_frame)
                self._timings_count += 1
                self._print_timing_summary(every=5000)
                n += 1
                pbar.update(1)

        self.video.release()

        df_tracked = track_objects(self.df, search_range=self.SEARCH_RANGE, memory=self.MEMORY)

        fps = self.config["FPS"]
        pogobot_diameter_cm = self.config["POGOBOT_DIAMETER_CM"]
        pixel_diameter = self.config["PIXEL_DIAMETER"]
        df_transformed = convert_datas(df_tracked, fps, pogobot_diameter_cm, pixel_diameter)

        df_transformed.to_csv(self.save_path, index=False)

        if bool(self.config.get("PLOT_TRAJECTORIES", False)):
            plot_trajectories(self.save_path, "Trajectories", self.config, bg_path=self.background_path)

        if bool(self.config.get("DEBUG_MODE", False)): #create a boolean for this !!!! plus, visualization parameters are WRONG for ESPCI arena
            pass
            #visualize_gif(self.save_path, self.config, self.background_path)

        end = time.time()
        print(f"Processed {n} frames in {round(end - start, 2)}s → {self.save_path}")