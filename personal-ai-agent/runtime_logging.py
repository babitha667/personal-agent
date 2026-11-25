"""Terminal activity and rotating local debug logs."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(project_dir: Path) -> Path:
    log_path = project_dir / "tmp" / "agent.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    ))
    logging.basicConfig(
        level=logging.DEBUG, handlers=[file_handler], force=True,
    )
    logging.getLogger(__name__).debug("Logging initialized: %s", log_path)
    return log_path
