import argparse
import os
from src.video_processing import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Pogotrack video processing pipeline")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--background", required=True, help="Path to background image (.bmp)")
    parser.add_argument("--output", required=True, help="Path to save output CSV")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file")
    parser.add_argument("--visualize", required=False, nargs="+", type=int,
                        help="Frame numbers to visualize contours for (space-separated)")
    args = parser.parse_args()

    # Ensure output directory exists (no-op if already exists)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    vp = VideoProcessor(
        video_path=args.video,
        background_path=args.background,
        save_path=args.output,
        config_path=args.config,
        frame_visualize=args.visualize  # <-- fixed name + comma
    )
    vp.process()

if __name__ == "__main__":
    main()
