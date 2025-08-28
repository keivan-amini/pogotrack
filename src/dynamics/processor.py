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

from .cleaning import (
    should_discard_trajectory,
    trim_leading_trailing,
    interpolate_missing,
    filter_wall_hit,
    )

from .physics import (
    compute_omega_fft,
    compute_omega_noise,
    compute_v_msd,
    compute_radius,
    compute_omega_ucm,
)

from .plotting import (
    plot_msd_all,
    plot_msd_trials,
    plot_quantity,
    plot_msd_grid,
    plot_circle_fit_grid,
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
    "MAX_FRACTION_SKIPPED": 0.3,
    "MAX_SKIPPED_FRAMES": 4,
    "FPS": 22.46,
    "POGOBOT_DIAMETER_CM": 4.975,
    "PIXEL_DIAMETER": 97.01,
    "MAX_TAU_SECONDS": 1.5,
    "TAUS_PERCENTAGE": 0.1,
    "T_SUBSET": 1,
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


    def extract_physics(self, pogobot: str = None,
                        plot: bool = False):
            
        """
        Extract relevant physical variables for all trials of a given pogobot,
        and save them into a single .csv file.

        Physical variables
        ------------------
            1) omega_fft   : angular velocity from FFT of theta.
            2) omega_noise : angular velocity from mean dθ/dt.
            3) v_msd       : linear velocity from MSD slope.
            4) R           : radius of curvature from circle fit.
            5) omega_ucm   : angular velocity assuming uniform circular motion.

        Parameters
        ----------
            pogobot (str):
                Folder name associated with a certain pogobot,
                example: 'pog_121'.
            plot (bool):
                boolean flag aimed at saving MSD and radius
                plots, by default in results/plots.
                Default is False.
        """

        pog_folder = os.path.join(self.folder_path, pogobot)
        if not os.path.exists(pog_folder):
            raise FileNotFoundError(f"No folder for pogobot {pogobot}")

        # --- Create results/plot/{pogobot} folder if plotting is enabled ---
        results_plot_dir = None
        base_dir = os.path.abspath(os.path.join(self.folder_path, "..", ".."))
        results_dir = os.path.join(base_dir, "results", "dynamics")
        if plot:
            results_plot_dir = os.path.join(results_dir, "plot", pogobot)
            os.makedirs(results_plot_dir, exist_ok=True)

        results = []
        all_trials_per_pwm = {}
        all_circles_per_pwm = {}

        for pwm in self.tested_pwm:
            trials_data = []  # collect MSD data for this pwm
            circle_trials_data = [] #collect R data for this pwm

            for trial in range(self.trials_per_pwm):
                csv_path = os.path.join(pog_folder, f"{pogobot}_{pwm}_{trial}.csv")
                if not os.path.exists(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)

                    # If plotting enabled, set paths for radius fit plot
                    results_r_path = None
                    if plot and results_plot_dir:
                        results_r_path = os.path.join(
                            results_plot_dir, f"{pogobot}_{pwm}_{trial}_r.png"
                        )

                    # Compute physical quantities
                    trial_data = compute_v_msd(df, self.max_tau_seconds,
                                            self.taus_percentage)
                    v_msd = trial_data["v_msd"]

                    omega_fft = compute_omega_fft(df)
                    omega_noise = compute_omega_noise(df)
                    
                    circle_trial_data =  compute_radius(df, self.t_subset,
                                                        results_r_path, pogobot,
                                                        pwm, trial)

                    R = circle_trial_data["R"]

                    omega_ucm = compute_omega_ucm(v_msd, R)

                    results.append({
                        "pwm": pwm,
                        "trial": trial,
                        "omega_fft": omega_fft,
                        "omega_noise": omega_noise,
                        "v_msd": v_msd,
                        "R": R,
                        "omega_ucm": omega_ucm
                    })

                    # Store MSD data for overlay plotting
                    trials_data.append(trial_data)
                    circle_trials_data.append(circle_trial_data)

                except Exception as e:
                    print(f"⚠️ Skipping {pogobot} pwm={pwm} trial={trial} due to error: {e}")
            

            all_trials_per_pwm[pwm] = trials_data
            all_circles_per_pwm[pwm] = circle_trials_data


            # --- After all trials for this PWM, plot overlay if requested ---
            if plot and results_plot_dir:
                combined_path = os.path.join(
                    results_plot_dir, f"{pogobot}_{pwm}_all_trials_msd"
                )
                plot_msd_trials(trials_data, pwm, combined_path)

        if plot and results_plot_dir:
            grid_path = os.path.join(results_plot_dir, f"{pogobot}_all_pwm_msd")
            all_msd_path = os.path.join(results_plot_dir, f"{pogobot}_all_msd")
            circle_grid_path = os.path.join(results_plot_dir, f"{pogobot}_all_pwm_R")
            plot_msd_grid(all_trials_per_pwm, save_path=grid_path)
            plot_msd_all(all_trials_per_pwm, save_path=all_msd_path)
            plot_circle_fit_grid(all_circles_per_pwm, circle_grid_path)

        # --- Save physics results for all pwm/trials ---
        if results:
            df_results = pd.DataFrame(results)
            save_path = os.path.join(results_dir, f"{pogobot}_physics.csv")
            df_results.to_csv(save_path, index=False)
            print(f"✅ Saved physics results for {pogobot} to {save_path}")
        else:
            print(f"⚠️ No results extracted for {pogobot}")


    def plot_physics(self, pogobot: str = None):

        """

        Given a pogobot (folder) name, plot the .csv
        dataset corresponding to the saved physical
        variables related to the characterization dynamics:

        | pwm | trial | omega_fft | omega_noise | v_msd |    R   | omega_ucm |
        |-----|-------|-----------|-------------|-------|--------|-----------|
        | 200 |   0   |   1.045   |    0.001    | 0.057 |  0.036 |   1.582   |
        | 200 |   1   |   2.091   |   -0.007    | 0.060 |  0.036 |   1.666   |
        | ... |  ...  |    ...    |     ...     |   ... |   ...  |    ...    |

        mainly as a function of the pwm. Plot both collective statistics
        and individual one, per pogobot.

        Save plots regarding:

            1) angular velocity from FFT of theta
                as a function of pwm.
            2) linear velocity from MSD slope as
                a function of pwm.
            3) radius of curvature from circle fit
                trajectory as a function of pwm.
            4) angular velocity of trajectory as
                a function of pwm.
        
        Parameters
        ----------
            pogobot (str):
                Folder name associated with a certain pogobot,
                example: 'pog_121'.
        """

        base_dir = os.path.abspath(os.path.join(self.folder_path, "..", ".."))
        csv_dir = os.path.join(base_dir, "results", "dynamics", pogobot + "_physics.csv")
        plot_path = os.path.join(base_dir, "results", "dynamics", "plot", pogobot, pogobot)
        try:
            df = pd.read_csv(csv_dir)
            for quantity in self.plotting.keys():
                save_path = f"{plot_path}_{quantity}"
                plot_quantity(df, pogobot, self.name_pogs, save_path, quantity, self.plotting)
        except Exception as e:
            print(f"⚠️ Skipping {pogobot} due to error: {e}")


# --- Commands directly launched from CLI --- #

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

    def extract(self, pogobot: str = None,
                plot: bool = False):

        """
        Extract only physical variables from .csv datasets,
        to characterize the pogobot dynamics.

        Parameters
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (extract all
                the pogobots' .csv generated files).
            plot (bool):
                boolean flag aimed at saving plots or not.
                Default is False.
        """
        pogs_to_run = [pogobot] if pogobot else self.video_map.keys()

        for pog in pogs_to_run:
            if pog not in self.video_map:
                raise ValueError(f"Pogobot {pog} not found in workspace")
            self.extract_physics(pog, plot)


    def plot(self, pogobot: str = None):

        """
        Launch all plot routines assoicated to
        pogobots' characterization dynamics.

        Parameters
        ----------
            pogobot (str):
                folder name associated with a certain pogobot,
                example: 'pog_121'. Default is None (extract all
                the pogobots' .csv generated files).
        """

        pogs_to_run = [pogobot] if pogobot else self.video_map.keys()
        for pog in pogs_to_run:
            if pog not in self.video_map:
                raise ValueError(f"Pogobot {pog} not found in workspace")
            self.plot_physics(pog)