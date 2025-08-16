import cv2
import yaml
import pandas as pd
import time
import numpy as np
from tqdm import tqdm

from .plot_helpers import debug_frame, visualize_contours, plot_trajectories
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


class VideoProcessor:

    DEFAULTS = {
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

        "PLOT_TRAJECTORIES": False,
        "ARENA_XLIM": [0, 2992],
        "ARENA_YLIM": [0, 2976],
        "ARROW_LENGTH_VIS": 100,
        "ARENA_RADIUS": 50,
        "HEAD_WIDTH": 30,
        "HEAD_LENGTH": 30,
        "TRAJECTORY_DPI": 300,
        "VISUALIZE_CONTOURS_FRAMES": [],
    }

    def __init__(self, video_path, background_path, save_path, config_path, frame_visualize=None):
        """
        Main class for processing Pogobot arena videos and exporting tracking data to CSV.

        Args:
            video_path (str): Path to the input video (.mp4).
            background_path (str): Path to the background image (.bmp).
            save_path (str): Path where the output CSV will be saved.
            config_path (str): Path to YAML configuration file.
            frame_visualize (int | list[int] | None): Frames for which to visualize contours.
        """
        self.video_path = video_path
        self.background_path = background_path
        self.save_path = save_path

        # Load YAML config
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

        # Merge defaults → YAML overrides
        self._load_config(yaml_config)

        # Normalize frames-to-visualize: merge CLI (if any) with YAML list
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
                    # Not iterable -> ignore
                    pass

        _add_frames(yaml_frames)
        _add_frames(cli_frames)

        # Results DF
        self.df = pd.DataFrame(columns=["frame", "x", "y", "theta"])

    def _load_config(self, yaml_config: dict):
        """Merge default values with YAML config, YAML has priority."""
        merged = {**self.DEFAULTS, **yaml_config}
        self.config = merged
        for key, value in merged.items():
            setattr(self, key, value)

    def _load_video_and_background(self):
        """Load video and background image."""
        self.video = cv2.VideoCapture(self.video_path)
        self.background = cv2.imread(self.background_path)
        self.background = cv2.flip(self.background, 0)

    def _create_mask(self):
        """Create circular mask for arena."""
        mask = np.zeros(self.background.shape[:2], dtype=np.uint8)
        cv2.circle(mask, self.CENTER, self.RADIUS, 255, -1)
        return mask

    def process(self):
        """Run the video processing pipeline."""
        start = time.time()
        self._load_video_and_background()
        mask = self._create_mask()

        total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        n = 0

        with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
            while self.video.isOpened():
                ret, frame = self.video.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 0)

                # Apply mask
                frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
                background_masked = cv2.bitwise_and(self.background, self.background, mask=mask)

                # Processing pipeline
                diff = get_difference(frame_masked, background_masked)
                thresh = binarize(diff, threshold=self.THRESHOLD)
                contours = find_contours(thresh, area_params=self.AREAS, peri_params=self.PERIMETERS)
                x, y = get_position(contours)
                thetas = get_all_angles(thresh, y, x)

                # Fallback if detection count mismatch
                if len(thetas) != self.N_POGO:
                    print(f"Frame {n}: {len(thetas)} bots detected. Trying fallback parameters...")

                    contours = find_contours(thresh, area_params=[1000, 10000], peri_params=[100, 900])
                    x, y = get_position(contours)
                    thetas = get_all_angles(thresh, y, x)

                    if len(thetas) != self.N_POGO:
                        print(f"Frame {n}: Still {len(thetas)} bots detected. Interrupting!")
                        if self.config.get("DEBUG_MODE", False):
                            debug_frame(frame_masked, diff, thresh, contours, title=f"Debug frame {n}")
                        break  # keep as-is per your current behavior

                # Visualize contours if requested for this frame index
                if n in self.frames_to_visualize:
                    visualize_contours(diff, contours, x, y, thetas, self.config)

                self.df = save_datas(self.df, n, x, y, thetas)
                n += 1
                pbar.update(1)

        self.video.release()

        # Track IDs
        df_tracked = track_objects(self.df, search_range=self.SEARCH_RANGE)

        # Convert to seconds/cm using params from config
        fps = self.config["FPS"]
        pogobot_diameter_cm = self.config["POGOBOT_DIAMETER_CM"]
        pixel_diameter = self.config["PIXEL_DIAMETER"]
        df_transformed = convert_datas(df_tracked, fps, pogobot_diameter_cm, pixel_diameter)

        # Save transformed CSV
        df_transformed.to_csv(self.save_path, index=False)

        # Optional trajectories plot (controlled from YAML)
        plot_trajectories(self.save_path, "Tracking Results", self.config, bg_path=self.background_path)

        end = time.time()
        print(f"Processed {n} frames in {round(end - start, 2)}s → {self.save_path}")
