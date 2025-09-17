import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    """
    Set up logging configuration using the LOG_LEVEL from application settings.
    """
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
