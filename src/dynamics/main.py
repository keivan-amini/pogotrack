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
    
    Example:

    /usr/bin/python3 -m src.dynamics.main \
    --video_dir data/tpu \
    --background data/tpu/bkg.bmp \
    --dynconfig config/dynamics.yaml \
    --processconfig config/default.yaml \
    --mode check \
    --pogobot pog_84

    To run in parallel from the terminal two pogobots:

    /usr/bin/python3 -m src.dynamics.main \
        --video_dir data/tpu \
        --background data/tpu/bkg.bmp \
        --dynconfig config/dynamics.yaml \
        --processconfig config/default.yaml \
        --mode process \
        --pogobot pog_91 &

    /usr/bin/python3 -m src.dynamics.main \
        --video_dir data/tpu \
        --background data/tpu/bkg.bmp \
        --dynconfig config/dynamics.yaml \
        --processconfig config/default.yaml \
        --mode process \
        --pogobot pog_95 &
    wait


    TODO  
    
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
    args = parser.parse_args()

    print("Launching DynamicsProcessor...")
    processor = DynamicsProcessor(
        folder_path = args.video_dir,
        background_path = args.background,
        dyn_config_path = args.dynconfig,
        processing_config_path = args.processconfig
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
        processor.extract(pogobot = args.pogobot) #TODO
    
    if args.mode == "plot":
        processor.plot(pogobot = args.pogobot) #TODO

    if args.mode == "complete":
        processor.run_all(pogobot = args.pogobot)


if __name__ == "__main__":
    main()
