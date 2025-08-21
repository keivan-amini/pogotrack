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
    }

    def __init__(self, folder_path, background_path, dyn_config_path):


        """
        Initialize class by defining the attribute variables.

        Parameters
        ----------
            video_path (str):
                path to the input dynamics experiment (.mp4).
            background_path (str):
                path to the background image (.bmp).
            config_path (str):
                path to dynamics parameter .yaml configuration file.
        """

        self.folder_path = folder_path
        self.background_path = background_path

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
            pog (str): 
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
        

    def process_runs(self, pog: str):

        """
        Run VideoProcessor on each trimmed video for one pogobot.
        Saves CSV next to video.

        Parameters
        ----------
            pog (str):
                pogobot' ID, e.g., "pog_121"

        """

        pog_folder = os.path.join(self.folder_path, pog)
        if not os.path.exists(pog_folder):
            raise FileNotFoundError(f"No folder for pogobot {pog}")

        for pwm in self.tested_pwm:
            for trial in range(self.trials_per_pwm):
                video_filename = f"{pog}_{pwm}_{trial}.mp4"
                video_path = os.path.join(pog_folder, video_filename)

                if not os.path.exists(video_path):
                    continue

                output_csv = video_path.replace(".mp4", ".csv")
                try:
                    vp = VideoProcessor(video_path, self.background_path)
                    vp.process(output_csv)
                    print(f"✅ CSV saved: {output_csv}")
                except Exception as e:
                    print(f"❌ Error processing {video_filename}: {e}")



    def run_all(self):
        
        """
        Run full pipeline: trim + process for all pogobots found.
        """
        
        for pog, video_path in self.video_map.items():
            self.trim_pogobot_runs(pog, video_path)
            self.process_runs(pog)

    def trim(self):

        """
        Trim only pogobot videos.
        """

        for pog, video_path in self.video_map.items():
            self.trim_pogobot_runs(pog, video_path)

    def process(self):

        """
        Process only pogobot videos, obtaining .csv datasets.
        """

        for pog in self.video_map:
            self.process_runs(pog)
    
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