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
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import (
    get_difference,
    binarize,
    find_contours,
    get_position,
    get_all_angles,
    track_objects,
    save_datas,
    convert_datas,
)

from .plot_helpers import (
    debug_frame,
    visualize_contours,
    plot_trajectories,
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
        "SEARCH_RANGE": 200,
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
    }

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

        self._load_config(yaml_config)

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
        self.config = merged
        for key, value in merged.items():
            setattr(self, key, value)

    def _load_video_and_background(self):

        """
        Load video and background image.
        """

        self.video = cv2.VideoCapture(self.video_path)
        self.background = cv2.imread(self.background_path)
        self.background = cv2.flip(self.background, 0)

    def _create_mask(self):

        """
        Create a circular binary mask for the arena.

        Return
        ------
        mask (np.ndarray):
            Binary mask with arena pixels set to 255 and
            outside pixels set to 0.
        """

        mask = np.zeros(self.background.shape[:2], dtype=np.uint8)
        cv2.circle(mask, self.CENTER, self.RADIUS, 255, -1)
        return mask

    def skip_frame(self, n: int):

        """
        Skip a frame during processing and insert
        placeholder data.

        This method records zeros for positions and
        orientations when detection fails, ensuring
        that the output dataframe preserves frame continuity.

        Parameters
        ----------
            n (int):
                index of frame to skip
        
        Return
        ------
            n (int):
                index of next frame to analyze
        """

        self.df = save_datas(self.df, n,[0],[0],[0])
        return n + 1

    def _adjust_detection(self, diff):

        """
        Attempt to recover correct Pogobot detections by
        tuning parameters.

        Strategy
        --------
        1. Sweep threshold around `self.THRESHOLD` (± up to MAX_OFFSET).
        2. If still unsuccessful, broaden area and perimeter ranges
        using `self.FALLBACK_AREA` and `self.FALLBACK_PERIMETERS`.
        3. Stop at the first successful configuration.

        Parameters
        ----------
        diff (np.ndarray):
            frame difference image after background
            subtraction.

        Returns
        -------
        contours (list):
            list of detected contours.
        x (list):
            Pogobots x-coordinates.
        y (list):
            Pogobots y-coordinates.
        thetas (list):
            pogobot orientation angles.
        success (bool):
            whether detection matched expected number of Pogobots.
        """

        base_thresh = self.THRESHOLD
        max_offset = self.MAX_OFFSET

        # 1. Try thresholds in both directions
        offsets = list(range(1, max_offset + 1))
        candidates = [base_thresh + o for o in offsets] + [base_thresh - o for o in offsets]

        for t in candidates:
            if t <= 0:
                continue
            thresh = binarize(diff, threshold=t)
            contours = find_contours(thresh, area_params=self.AREAS, peri_params=self.PERIMETERS)
            x, y = get_position(contours)
            thetas = get_all_angles(thresh, y, x)
            if len(thetas) == self.N_POGO:
                print("Success.")
                return contours, x, y, thetas, True

        # 2. Try wider area/perimeter ranges
        thresh = binarize(diff, threshold=base_thresh)
        contours = find_contours(thresh, area_params=self.FALLBACK_AREA, peri_params=self.FALLBACK_PERIMETERS)
        x, y = get_position(contours)
        thetas = get_all_angles(thresh, y, x)

        if len(thetas) == self.N_POGO:
            print("Success.")
            return contours, x, y, thetas, True

        # 3. If nothing worked, return last attempt
        return contours, x, y, thetas, False


    def process(self):

        """
        Run the full video processing pipeline.

        Steps
        -----
        1. Load video and background.
        2. Apply arena mask.
        3. For each frame:
        - Compute difference with background.
        - Apply threshold and contour detection.
        - Extract Pogobot positions and orientations.
        - Apply fallback strategy if detections mismatch.
        - Save detections to dataframe.
        4. Track Pogobots across frames and assign IDs.
        5. Convert pixels to centimeters and frames to seconds.
        6. Save trajectories to CSV.
        7. Optionally plot trajectories.

        Notes
        -----
        - Frames where detection fails are skipped with placeholder data.
        - Debug visualization is controlled by the `DEBUG_MODE` config flag.
        - Fallback parameters are defined by `MAX_OFFSET`, `FALLBACK_AREA`,
        and `FALLBACK_PERIMETERS`.

        Output
        ------
        - CSV file containing tracked and transformed Pogobot trajectories.
        - Optional trajectory plots if enabled in config.
        """

        start = time.time()
        self._load_video_and_background()
        mask = self._create_mask()

        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) 
        n = 0

        with tqdm(total = total_frames, desc= "Processing video", unit= "frame") as pbar:
            while self.video.isOpened():
                ret, frame = self.video.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 0)

                # Apply mask
                frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
                background_masked = cv2.bitwise_and(self.background, self.background, mask = mask)

                # Processing pipeline
                diff = get_difference(frame_masked, background_masked)
                thresh = binarize(diff, threshold = self.THRESHOLD)
                contours = find_contours(thresh, area_params=self.AREAS, peri_params=self.PERIMETERS)
                x, y = get_position(contours)
                thetas = get_all_angles(thresh, y, x)

                if len(thetas) != self.N_POGO:
                    print(f"Frame {n}: {len(thetas)} bots detected. Attempting fallback...")
                    contours, x, y, thetas, success = self._adjust_detection(diff)

                    if not success:
                        print(f"Frame {n}: Still {len(thetas)} bots detected. Skipping frame.")
                        if bool(self.config.get("DEBUG_MODE", False)):
                            debug_frame(frame_masked, diff, thresh, contours, title=f"Debug frame {n}")
                        n = self.skip_frame(n)
                        pbar.update(1)
                        continue

                # Visualize contours if requested for this frame index
                if n in self.frames_to_visualize:
                    visualize_contours(diff, contours, x, y, thetas, self.config)

                self.df = save_datas(self.df, n, x, y, thetas)
                n += 1
                pbar.update(1)

        self.video.release()

        # Track IDs
        df_tracked = track_objects(self.df, search_range = self.SEARCH_RANGE)

        # Convert to seconds/cm using params from config
        fps = self.config["FPS"]
        pogobot_diameter_cm = self.config["POGOBOT_DIAMETER_CM"]
        pixel_diameter = self.config["PIXEL_DIAMETER"]
        df_transformed = convert_datas(df_tracked, fps, pogobot_diameter_cm, pixel_diameter)

        df_transformed.to_csv(self.save_path, index = False)

        # Optional trajectories plot (controlled from .yaml)
        if bool(self.config.get("PLOT_TRAJECTORIES", False)):
            plot_trajectories(self.save_path, "Trajectories", self.config, bg_path = self.background_path)

        end = time.time()
        print(f"Processed {n} frames in {round(end - start, 2)}s → {self.save_path}")
