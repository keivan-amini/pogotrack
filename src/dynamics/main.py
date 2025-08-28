"""
Main script controlling the execution of the dynamics
characterization pipeline, through a CLI.
"""

import argparse
from .processor import DynamicsProcessor


def main():

    """
    Description
    -----------

    The execution of this function launches
    the dynamics processing pipeline, i.e.,
    the extraction of dynamical variables such
    as the linear velocity, the angular velocity,
    or the curvature radius given a .mp4 video
    dynamical experiment in input.

    A pogobot dynamical experiment in this context
    is defined as a video where, given a selected list
    of TESTED_PWM values, a pogobot runs for RUN_DURATION
    seconds, then stops for PAUSE_DURATION seconds and repeat
    the process for TRIALS_PER_PWM times. Then, it repeat the
    process for the next value of TESTED_PWM list.

    All the cited parameters are tunable in config/dynamics.yaml.
    
    Example:

    /usr/bin/python3 -m src.dynamics.main \
        --video_dir data/tpu \
        --background data/tpu/bkg.bmp \
        --dynconfig config/dynamics.yaml \
        --processconfig config/default.yaml \
        --mode extract --plot \
        --pogobot pog_88
    
    This example command will automatically launch the
    dynamics characterization in the data/tpu folder, where
    it will be expected to have at least one 'pog_id.mp4' video
    figuring one pogobot dynamical experiment. Video will be
    processed using a background image data/tpu/bkg.bmp in this
    case, using the config/dynamics.yaml dynamical parameters
    and the config/default.yaml video processing parameters.
    At the moment we can launch different steps of the dynamics
    pipeline, which in order are:

        1) --mode trim
            this command cuts 'pog_id.mp4' in several different
            .mp4 videos where pog is supposed to run, inside the
            folder 'pog_id', using the video cutting dynamics
            parameters. Videos are saved as: pog_id_pwm_trial.mp4.

        2) --mode process
            this command processes the generated video and obtains
            a pogobot .csv file with features 'time, x, y, theta, id'
            in path pog_id/pog_id_pwm_trial.csv.

        3) --mode check
            this command just checks that all the videos have been
            processed and it is possible to access all the generated
            .csv file.

        4) --mode clean
            this command clean the generated .csv datasets, mainly
            removing datasets with too many missing rows (problems during
            video processing), trim leading and trailing skipped frames,
            interpolating isolated skipped frames and filtering outer
            trajectories (when pogobots hit walls).

        5) --mode extract (optional) --plot
            this command extract dynamical variables from the cleaned
            generated .csv dataset, namely: angular velocity, linear
            velocity, radius of curvature, and angular velocity
            assuming uniform circular motion. 
            Datasets are saved in results/dynamics/pog_id_physics.csv.
            The --plot flag allows us to save grids per-pwm plots for
            each pogobot, in results/dynamics/plot/pog_id folder.

        6) --mode plot
            save plots regarding the extracted dynamical variables
            as a function of the pwm in results/dynamics/plot/pog_id 
            folder.
        
        *7) --mode complete
            launch each dynamical processing step sequentially
            from 1 to 6 (not recommended)
    
    """

    parser = argparse.ArgumentParser(description = "Dynamics characterization pipeline")
    parser.add_argument("--video_dir", required = True, help = "Folder with pogobot videos")
    parser.add_argument("--background", required = True, help = "Path to background image (.bmp)")
    parser.add_argument("--dynconfig", required = True, help = "Path to dynamics .yaml config")
    parser.add_argument("--processconfig", required = True, help = "Path to processing .yaml config")
    parser.add_argument("--mode", required = True, type = str,
                        help = "Which step of the dynamics characterization to launch." \
                        "Possible arguments: trim, process, check, clean, extract, plot, or complete")
    parser.add_argument("--pogobot", required = False, type = str,
                        help = "Focus only on one pogobot! Example: pog_191")
    parser.add_argument("--plot", action = "store_true", help = "Enable plotting when using mode = extract")
    args = parser.parse_args()

    if args.plot and args.mode != "extract":
        parser.error("--plot can only be used when --mode = extract")

    print("Launching DynamicsProcessor...")
    processor = DynamicsProcessor(
        folder_path = args.video_dir,
        background_path = args.background,
        dyn_config_path = args.dynconfig,
        processing_config_path = args.processconfig,
    )

    if args.mode == "trim":
        processor.trim(pogobot = args.pogobot)

    if args.mode == "process":
        processor.process(pogobot = args.pogobot)
    
    if args.mode == "check":
        processor.check(pogobot = args.pogobot)
    
    if args.mode == "clean":
        processor.clean(pogobot = args.pogobot)
    
    if args.mode == "extract":
        processor.extract(pogobot = args.pogobot, plot = args.plot)
    
    if args.mode == "plot":
        processor.plot(pogobot = args.pogobot)

    if args.mode == "complete":
        processor.run_all(pogobot = args.pogobot)


if __name__ == "__main__":
    main()
