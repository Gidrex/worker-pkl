"""Main entry point for parking detection system."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from core.frame_processor import FrameProcessor
from storage.database import Database
from utils.config import load_config
from utils.logging import setup_logging


def process_video_command(args: argparse.Namespace) -> int:
    """Process video file command.

    Args:
        args: Command arguments

    Returns:
        Exit code
    """
    config = load_config(args.config)
    setup_logging(config.logging)

    logger.info("Starting parking detection system")

    database = Database(config.storage.database_path)
    processor = FrameProcessor(config, database)

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {args.video}")
        return 1

    video_id = processor.process_video(str(video_path))

    logger.success(f"Video processed successfully: video_id={video_id}")

    return 0


def init_config_command(args: argparse.Namespace) -> int:
    """Initialize default config file.

    Args:
        args: Command arguments

    Returns:
        Exit code
    """
    from utils.config import Config, save_config

    config = Config()
    save_config(config, args.config)

    logger.info(f"Created default config: {args.config}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parking Vehicle Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config file (default: config.json)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    process_parser = subparsers.add_parser("process", help="Process video file")
    process_parser.add_argument("video", help="Path to video file")

    subparsers.add_parser("init", help="Initialize default config")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "process":
        return process_video_command(args)

    if args.command == "init":
        return init_config_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
