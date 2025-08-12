import argparse
import os
from src.video_processing import VideoProcessor


def main():
    parser = argparse.ArgumentParser(description="Pogotrack video processing pipeline")
    parser.add_argument("--video", required=True, help="Path to input video (.mp4)")
    parser.add_argument("--background", required=True, help="Path to background image (.bmp)")
    parser.add_argument("--output", required=True, help="Path to save output CSV")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config file")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Process video
    vp = VideoProcessor(
        video_path=args.video,
        background_path=args.background,
        save_path=args.output,
        config_path=args.config
    )
    vp.process()


if __name__ == "__main__":
    main()
