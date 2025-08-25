"""
This module has the aim to construct the DynamicsProcessor class,
useful to launch the dynamics characterization of pogobots running
in an experimental arena.

To properly use this class, it is necessary to perform one (or multiple)
experiment facing one pogobot running with multiples pre-defined PWM
for a pre-defined number of time. Then, this class trims and batches
the whole video and obtain different dataset for each run.
"""

## at the moment focusing on a situation where just one pogobot is in the arena
## implement CLI !

import os
import yaml
import ffmpeg
import pandas as pd

from .utils import (
    should_discard_trajectory,
    trim_leading_trailing,
    interpolate_missing,
    filter_wall_hit,
    )

from ..video_processing import VideoProcessor


class DynamicsProcessor:
    """
    Description
    -----------
    Handles extraction and processing of videos for dynamics characterization.
    - Prepares workspace (one folder per pogobot)
    - Trims videos into runs (per PWM, per trial)
    - Processes runs with VideoProcessor → saves per-run CSV


    Methods
    -------
    TODO

    Attributes
    ----------
    TODO
    """

    DEFAULT_DYNAMICS = {
    "NAME_POGS": ["pog_191", "pog_192", "pog_193", "pog_194", "pog_195"],  
    "N_POGO": 1, 
    "RUN_DURATION": 10,
    "PAUSE_DURATION": 7.68,
    "DEFAULT_START_TIME": 3,
    "TESTED_PWM": [500, 600, 650, 700, 750, 800, 850, 900, 950, 1023],
    "TRIALS_PER_PWM": 10,
    "MIN_FRAMES": 100,
    "CENTER_X": 1518,
    "CENTER_Y": 1529,
    "OUTER_RADIUS": 1200,
    "MIN_SECONDS": 4,
    "MAX_FRACTION_SKIPPED": 0.3,
    "MAX_SKIPPED_FRAMES": 4,
    }

    def __init__(self, folder_path, background_path, dyn_config_path, processing_config_path):


        """
        Initialize class by defining the attribute variables.

        Parameters
        ----------
            video_path (str):
                path to the input dynamics experiment (.mp4).
            background_path (str):
                path to the background image (.bmp).
            dyn_config_path (str):
                path to dynamics parameter .yaml configuration file.
            processing_config_path (str):
                path to processing parameter .yaml configuration file.
        """

        self.folder_path = folder_path
        self.background_path = background_path
        self.processing_config = processing_config_path

        yaml_config = {}
        if dyn_config_path and os.path.exists(dyn_config_path):
            with open(dyn_config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

        self._load_config(yaml_config)
        self.video_map = self.prepare_workspace(self.folder_path)



    def _load_config(self, yaml_config: dict):

        """
        Merge default processing parameters with .yaml configuration.
        .yaml has the priority.

        Parameters
        ----------
            yaml_config : dict
                dictionary of configuration parameters loaded from .yaml.
        """

        normalized_yaml = {k.upper(): v for k, v in (yaml_config or {}).items()}
        merged = {**self.DEFAULT_DYNAMICS, **normalized_yaml}
        self.config = merged
        # expose lowercase attributes
        for key, value in merged.items():
            setattr(self, key.lower(), value)


    def prepare_workspace(self, folder_path):
        """
        Ensure workspace structure: each pogobot has its own folder.

        Parameters
        ----------
            folder_path (str)
                path where .mp4 videos are expected.
        
        Return
        ------
            video_map (dict):
                dict containing as keys the pogobot ID, and as
                values the video path associated to the pogobot.

        """

        video_map = {}
        for pog in self.name_pogs:
            video_filename = f"{pog}.mp4"
            video_path = os.path.join(folder_path, video_filename)

            if not os.path.exists(video_path):
                print(f"⚠️ Missing video for {pog}: {video_filename}")
                continue

            pog_folder = os.path.join(folder_path, pog)
            os.makedirs(pog_folder, exist_ok=True)

            video_map[pog] = video_path
            print(f"📂 Workspace ready for {pog}: {pog_folder}")

        return video_map



    def trim_pogobot_runs(self, pog: str, input_video_path: str):

        """
        Cuts a long test video into smaller clips (per PWM, per trial).
        Saved as: {pog}/{pog}_{pwm}_{trial}.mp4

        Parameters
        ----------
            pogobot (str): 
                pogobot' ID, e.g., "pog_121".
            input_video_path (str):
                full video path for video to be trimmed.

        """

        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video not found: {input_video_path}")

        print(f"🎬 Processing video for {pog}: {input_video_path}")

        for pwm_index, pwm in enumerate(self.tested_pwm):
            for trial in range(self.trials_per_pwm):
                run_index = pwm_index * self.trials_per_pwm + trial
                clip_start = self.default_start_time + run_index * (
                    self.run_duration + self.pause_duration
                )

                output_name = f"{pog}_{pwm}_{trial}.mp4"
                output_path = os.path.join(self.folder_path, pog, output_name)

                try:
                    (
                        ffmpeg
                        .input(input_video_path, ss=clip_start, t=self.run_duration)
                        .output(output_path, codec="copy")
                        .run(overwrite_output=True, quiet=True)
                    )
                    print(f"✅ Created: {output_path}")
                except ffmpeg.Error as e:
                    print(f"❌ Error trimming run {run_index} ({pwm}, trial {trial}): {e}")
        

    def process_runs(self, pogobot: str):

        """
        Run VideoProcessor on each trimmed video for one pogobot.
        Saves CSV next to video.

        Parameters
        ----------
            pogobot (str):
                pogobot' ID, e.g., "pog_121"

        """

        pog_folder = os.path.join(self.folder_path, pogobot)
        if not os.path.exists(pog_folder):
            raise FileNotFoundError(f"No folder for pogobot {pogobot}")

        for pwm in self.tested_pwm:
            for trial in range(self.trials_per_pwm):
                video_filename = f"{pogobot}_{pwm}_{trial}.mp4"
                video_path = os.path.join(pog_folder, video_filename)

                if not os.path.exists(video_path):
                    continue

                output_csv = video_path.replace(".mp4", ".csv")
                try:
                    vp = VideoProcessor(video_path, self.background_path, output_csv, self.processing_config)
                    vp.process()
                    print(f"✅ CSV saved: {output_csv}")
                except Exception as e:
                    print(f"❌ Error processing {video_filename}: {e}")

    def check_csv(self, pogobot: str):

        """
        In the generated .csv dataset, print only the last
        element of the column [time]. Useful for debugging
        purposes.

        Parameters
        ----------
            pogobot (str):
                pogobot' ID, e.g., "pog_121"

        """

        pog_folder = os.path.join(self.folder_path, pogobot)
        if not os.path.exists(pog_folder):
            raise FileNotFoundError(f"No folder for pogobot {pogobot}")

        for pwm in self.tested_pwm:
            for trial in range(self.trials_per_pwm):
                video_filename = f"{pogobot}_{pwm}_{trial}.mp4"
                video_path = os.path.join(pog_folder, video_filename)

                if not os.path.exists(video_path):
                    continue

                output_csv = video_path.replace(".mp4", ".csv")
                try:
                    df = pd.read_csv(output_csv)
                    last_time = df["time"].iloc[-1]
                    if last_time < self.min_seconds: #UPDATE: with the video porcessing skipping frame procedure, this does not make sense anymore
                        print(f"❌ Problems found with the following video: {video_path}: {round(last_time, 2)} s")
                except:
                    print(f"❌ Problems found with the following video: {output_csv}: not found.")

    def clean_csv(self, pogobot: str = None):

        """
        Clean the generated .csv dataset:
            1) Remove datasets with too many missing rows.
            2) Trim leading/trailing skipped frames.
            3) Interpolate isolated skipped frames.
            4) Filter outer trajectories (Pogobot hitting wall).
        
        Parameters:
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (clean all
                the pogobots' .csv generated files)    
        """


        pog_folder = os.path.join(self.folder_path, pogobot)
        if not os.path.exists(pog_folder):
            raise FileNotFoundError(f"No folder for pogobot {pogobot}")

        for pwm in self.tested_pwm:
            for trial in range(self.trials_per_pwm):
                csv_path = f"{pogobot}_{pwm}_{trial}.csv"
                csv_path = os.path.join(pog_folder, csv_path)
                if not os.path.exists(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    print(f"\n📂 Processing {csv_path} (raw={len(df)} rows)")

                    # (1) Trim edges first so they never influence discard logic
                    df = trim_leading_trailing(df)
                    print(f"   → after trim_leading_trailing: {len(df)} rows")
                    if df.empty:
                        print(f"🗑️ Discarded {csv_path} (empty after edge trim)")
                        os.remove(csv_path)
                        continue

                    # (2) Decide if this trajectory should be discarded
                    keep = should_discard_trajectory(
                        df,
                        max_fraction=self.max_fraction_skipped,
                        max_skipped=self.max_skipped_frames
                    )
                    if keep is None:
                        print(f"🗑️ Discarded {csv_path} (too many missing frames)")
                        os.remove(csv_path)
                        continue
                    print(f"   → after should_discard_trajectory: {len(df)} rows")

                    # (3) Interpolate isolated interior gaps
                    df = interpolate_missing(df)
                    print(f"   → after interpolate_missing: {len(df)} rows")

                    # (4) Trim after a wall hit
                    df = filter_wall_hit(df, self.center_x, self.center_y, self.outer_radius,
                                         self.pogobot_diameter_cm, self.pixel_diameter)
                    print(f"   → after filter_wall_hit: {len(df)} rows")

                    # Final guard
                    if df is None or df.empty:
                        print(f"🗑️ Discarded {csv_path} (empty after cleaning)")
                        os.remove(csv_path)
                        continue

                    # Persist cleaned data
                    df.to_csv(csv_path, index=False)
                    print(f"✅ Cleaned {csv_path} (final={len(df)} rows)")

                except Exception as e:
                    print(f"❌ Problem reading {csv_path}: {e}")

    def run_all(self, pogobot: str = None):
        
        """
        Run full pipeline: trim + process for all pogobots found.
        If pogobot is provided, only run the full pipeline for him.

        Parameters
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (run full
                pipeline for all the pogobots' videos)
        """
        
        pogs_to_run = [pogobot] if pogobot else self.video_map.keys()

        for pog in pogs_to_run:
            if pog not in self.video_map:
                raise ValueError(f"Pogobot {pog} not found in workspace")

            video_path = self.video_map[pog]
            self.trim_pogobot_runs(pog, video_path)
            self.process_runs(pog) # add the new methods: check, clean, extract, plot

    def trim(self, pogobot: str = None):

        """
        Trim only pogobot videos.
        If pogobot is provided, only trim his associated video.

        Parameters
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (trim all
                the pogobots' videos)

        """

        pogs_to_run = [pogobot] if pogobot else self.video_map.keys()

        for pog in pogs_to_run:
            if pog not in self.video_map:
                raise ValueError(f"Pogobot {pog} not found in workspace")
            self.trim_pogobot_runs(pog, self.video_map[pog])

    def process(self, pogobot: str = None):

        """
        Process only pogobot videos, obtaining .csv datasets.
        If pogobot is provided, only process his associated video.

        Parameters
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (process all
                the pogobots' videos)

        """

        pogs_to_run = [pogobot] if pogobot else self.video_map.keys()

        for pog in pogs_to_run:
            if pog not in self.video_map:
                raise ValueError(f"Pogobot {pog} not found in workspace")
            self.process_runs(pog)

    def check(self, pogobot: str = None):

        """
        In the generated .csv dataset, print only the last
        element of the column [time]. Useful for debugging
        purposes.

        Parameters
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (check all
                the pogobots' .csv generated files)

        """
        pogs_to_run = [pogobot] if pogobot else self.video_map.keys()

        for pog in pogs_to_run:
            if pog not in self.video_map:
                raise ValueError(f"Pogobot {pog} not found in workspace")
            self.check_csv(pog)

    def clean(self, pogobot: str = None):

        """
        Launch only the clean_csv() function to the generated
        dataset.

        Parameters
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (clean all
                the pogobots' .csv generated files)

        """

        pogs_to_run = [pogobot] if pogobot else self.video_map.keys()

        for pog in pogs_to_run:
            if pog not in self.video_map:
                raise ValueError(f"Pogobot {pog} not found in workspace")
            self.clean_csv(pog)

    def extract(self):

        """
        Extract only physical variables from .csv datasets.
        """
        # TODO
        pass

    def plot(self):

        """
        Plot only.
        """

        # TODO
        pass