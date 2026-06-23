import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger for the application.

    Call once at process startup (e.g. from main.py) before any log
    messages are emitted.  Suppresses noisy third-party loggers.
    """
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Intentionally does not add handlers — callers depend on configure_logging()
    or pytest's log capture to install them.  Adding handlers here causes
    duplicate output when modules are imported before configure_logging runs.
    """
    return logging.getLogger(name)
