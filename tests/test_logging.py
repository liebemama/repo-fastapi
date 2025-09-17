from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import get_settings


class StartsWithFilter(logging.Filter):
    """
    Allows only log records whose logger names start with the given prefix (e.g., "plugins.").

    Attributes:
        prefix (str): The required prefix for logger names.
    """

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self.prefix)


def _level(name: str, default: int) -> int:
    """
    Convert a log level name to its corresponding logging module constant.

    Args:
        name (str): The log level name (e.g., "INFO", "DEBUG").
        default (int): The default logging level to use if conversion fails.

    Returns:
        int: The logging level constant.
    """
    try:
        return getattr(logging, str(name).upper())
    except Exception:
        return default


def setup_logging() -> None:
    """
    Configure logging: console output and optional rotating log files.

    Uses settings from the environment via the get_settings function.
    Sets up:
      1. Console logging for root logger.
      2. Error log file (if enabled).
      3. Plugins-specific log file (if enabled).
      4. Specific logger levels.
    """
    s = get_settings()

    # Console handler (root)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(_level(s.LOG_LEVEL, logging.INFO))
    console.setFormatter(logging.Formatter(s.LOG_CONSOLE_FORMAT))

    root = logging.getLogger()
    root.setLevel(_level(s.LOG_LEVEL, logging.INFO))
    root.handlers.clear()
    root.addHandler(console)

    # Error log file (ERROR+) if enabled
    if s.LOG_ERRORS_TO_FILE:
        err_path: Path = s.ERROR_LOG_FILE
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_fh = RotatingFileHandler(
            err_path,
            maxBytes=s.ERROR_LOG_MAX_BYTES,
            backupCount=s.ERROR_LOG_BACKUPS,
            encoding="utf-8",
        )
        err_fh.setLevel(logging.ERROR)
        err_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(err_fh)

    # Plugins log file (plugins.* only) if enabled
    if s.LOG_PLUGINS_TO_FILE:
        pl_path: Path = s.PLUGINS_LOG_FILE
        pl_path.parent.mkdir(parents=True, exist_ok=True)
        pl_fh = RotatingFileHandler(
            pl_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        pl_fh.setLevel(_level(s.LOG_LEVEL_PLUGINS, logging.INFO))
        pl_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        pl_fh.addFilter(StartsWithFilter("plugins"))
        root.addHandler(pl_fh)

    # Specific logger configurations
    logging.getLogger("errors").setLevel(logging.ERROR)
    logging.getLogger("uvicorn.access").setLevel(_level(s.LOG_LEVEL_UVICORN, logging.WARNING))


def main() -> None:
    setup_logging()
    logger = logging.getLogger("test_logger")
    plugin_logger = logging.getLogger("plugins.test_plugin")

    logger.debug("This is a DEBUG message from test_logger.")
    logger.info("This is an INFO message from test_logger.")
    logger.warning("This is a WARNING message from test_logger.")
    logger.error("This is an ERROR message from test_logger.")

    plugin_logger.debug("This is a DEBUG message from plugins.test_plugin.")
    plugin_logger.info("This is an INFO message from plugins.test_plugin.")
    plugin_logger.warning("This is a WARNING message from plugins.test_plugin.")
    plugin_logger.error("This is an ERROR message from plugins.test_plugin.")


if __name__ == "__main__":
    main()
