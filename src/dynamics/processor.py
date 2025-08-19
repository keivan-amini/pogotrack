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
    Wraps cutting (per PWM run) and batch processing with VideoProcessor.


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

        # really not sure if folder path should be an argument of the class !

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
        self.prepare_workspace(self.video_path)



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


    def prepare_workspace(self, folder_path):
        """
        Ensure workspace structure: each pogobot has its own folder.

        Parameters
        ----------
        folder_path (str)
            Path where .mp4 videos are expected.

        """
        video_map = {}

        for pog in self.name_pogs:
            video_filename = f"{pog}.mp4"
            video_path = os.path.join(folder_path, video_filename)

            if not os.path.exists(video_path):
                print(f"⚠️ Warning: missing video for {pog}: {video_filename}")
                continue

            pog_folder = os.path.join(folder_path, pog)
            os.makedirs(pog_folder, exist_ok=True)

            video_map[pog] = video_path
            print(f"📂 Workspace ready for {pog}: {pog_folder}")



    def extract_pogobot_runs(self, folder_path: str, video_filename: str = None):
        """
        Cuts a long test video into smaller clips (per PWM, per trial).

        Parameters
        ----------
            folder_path (str): 
                directory where the full video is located.
            video_filename (str):
                optional video filename (defaults to folder_name.mp4).
        """
        folder_name = os.path.basename(folder_path)

        if video_filename is None:
            video_filename = f"{folder_name}.mp4"

        input_video_path = os.path.join(folder_path, video_filename)

        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Video not found: {input_video_path}")

        print(f"🎬 Processing video: {input_video_path}")

        for pwm_index, pwm in enumerate(self.tested_pwm):
            pwm_folder = os.path.join(folder_path, str(pwm))
            os.makedirs(pwm_folder, exist_ok=True)

            for trial in range(self.trials_per_pwm):
                run_index = pwm_index * self.trials_per_pwm + trial
                clip_start = self.default_start_time + run_index * (
                    self.run_duration + self.pause_duration
                )
                clip_output_name = f"run_{trial}.mp4"
                output_path = os.path.join(pwm_folder, clip_output_name)

                try:
                    (
                        ffmpeg
                        .input(input_video_path, ss=clip_start, t=self.run_duration)
                        .output(output_path, codec="copy")  # no re-encoding
                        .run(overwrite_output=True, quiet=True)
                    )
                    print(f"✅ Created: {output_path}")
                except ffmpeg.Error as e:
                    print(f"❌ Error processing run {run_index} at PWM {pwm}: {e}")
        

    def batch_process_combined_videos(self, folder_path: str):
        """
        Process each trimmed video of a multi-Pogobot experiment,
        extract data per Pogobot, and save results in subfolders.

        Parameters
        --------
            folder_path (str): Root folder containing PWM subfolders with videos.
            background_path (str): Path to arena background image.
        """
        for pwm in self.tested_pwm:
            pwm_folder = os.path.join(folder_path, str(pwm))

            for trial in range(self.trials_per_pwm):
                video_filename = f"run_{trial}.mp4"
                video_path = os.path.join(pwm_folder, video_filename)

                if not os.path.exists(video_path):
                    print(f"⚠️ Missing: {video_path}")
                    continue

                # Temporary output before splitting particles
                temp_csv_path = os.path.join(folder_path, f"temp_{pwm}_{trial}.csv")

                try:
                    # Run arena analyzer (from video_processing pipeline)
                    vp = VideoProcessor(video_path, self.background_path)
                    vp.process(temp_csv_path)

                    # Load results
                    df = pd.read_csv(temp_csv_path)

                    if "particle" not in df.columns:
                        print(f"❌ No 'particle' column in {video_filename}")
                        continue

                    # Save each particle's data separately
                    for particle_id in df["particle"].unique():
                        df_particle = df[df["particle"] == particle_id].copy()

                        pog_folder = os.path.join(folder_path, f"pog_{particle_id}")
                        os.makedirs(pog_folder, exist_ok=True)

                        output_filename = f"pog_{particle_id}_{pwm}_{trial}.csv"
                        output_path = os.path.join(pog_folder, output_filename)

                        df_particle.to_csv(output_path, index=False)
                        print(f"✅ Saved: {output_path}")

                    os.remove(temp_csv_path)

                except Exception as e:
                    print(f"❌ Error in {video_filename}: {e}")
