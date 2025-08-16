"""
Main script controlling the execution of the source modules
through a CLI, launching the whole pogotrack pipeline.
"""

import argparse
import os
from src.video_processing import VideoProcessor

def main():

    """
    Description
    -----------
    
    The execution of this function launches the 
    whole pogotrack pipeline, i.e. the extraction
    of x, y, theta and id of a video recording
    pogobots running in an experimental arena.

    To correctly run this program, the .mp4 input
    video must contains pogobot running with the
    exoskeleton contained in /stl subfolder.

    Example:

    /usr/bin/python3 -m scripts.main \                
        --video data/example.mp4 \
        --background data/bkg.bmp \
        --output results/tracking.csv \
        --config config/default.yaml \
        --visualize 1 10 200

    will automatically launch the video analysis pipeline on
    example.mp4, using bkg.bmp as background image, using the
    parameters contained in config/default.yaml.
    The result, i.e. the dataset containing the most relevant
    physical variables characterizing each pogobot at each
    timestep, are saved in results/tracking.csv.

    The optional argument --visualize allow to visualize the
    extracted contours and theta angle for each pogobot, in 
    this case for frame 1, 10 and 200 of the video example.mp4.

    """

    parser = argparse.ArgumentParser(description = "Pogotrack video processing pipeline")
    parser.add_argument("--video", required = True, help = "Path to input video (.mp4)")
    parser.add_argument("--background", required = True, help = "Path to background image (.bmp)")
    parser.add_argument("--output", required = True, help = "Path to save output CSV")
    parser.add_argument("--config", default = "config/default.yaml", help = "Path to YAML config file")
    parser.add_argument("--visualize", required = False, nargs = "+", type = int,
                        help = "Frame numbers to visualize contours for (space-separated)")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    vp = VideoProcessor(
        video_path = args.video,
        background_path = args.background,
        save_path = args.output,
        config_path = args.config,
        frame_visualize = args.visualize 
    )
    vp.process()

if __name__ == "__main__":
    main()
