"""Logging configuration using loguru."""

import sys

from loguru import logger

from src.utils.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure loguru logger.

    Args:
        config: Logging configuration
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format=config.format,
        level=config.level,
        colorize=True,
    )

    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        format=config.format,
        level=config.level,
        rotation=config.rotation,
        retention="30 days",
        compression="zip",
    )

    logger.info(f"Logging initialized with level: {config.level}")
