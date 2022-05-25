import inspect
import logging.config
import types
from typing import Any, Dict, Optional, cast

DEFAULT_LOGGING_SCHEMA = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s]::%(levelname)s::%(name)s.%(funcName)s: %(message)s"
            "\n",
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },  # root logger
    },
}


def set_logging_config(logging_schema: Optional[Dict[str, Any]] = None) -> None:
    """Set base configurations for logging.

    Args:
        logging_schema: dict with configurations for loggers.

    """
    logging.config.dictConfig(logging_schema or DEFAULT_LOGGING_SCHEMA)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get or create a logger.

    It uses the context (class or method) to define a title for the logging messages.

    Args:
        name: name to use on logging.getLogger

    Returns:
        newly created or past defined logger.

    """
    if not name:
        outer_frame = cast(
            types.FrameType, cast(types.FrameType, inspect.currentframe()).f_back
        )
        module_name = outer_frame.f_globals.get("__name__")
        self_context = outer_frame.f_locals.get("self")
        caller_name = (
            f"{module_name}.{self_context.__class__.__name__}"
            if self_context
            else module_name
        )
        return logging.getLogger(caller_name)
    return logging.getLogger(name)
