import enum
import logging
from typing import Union

import coloredlogs

LOGGER_NAME = "meshcat_viz"


class LoggingLevel(enum.IntEnum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def getLogger(name: str = LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name=name)


def set_logging_level(level: Union[int, LoggingLevel] = LoggingLevel.WARNING):
    if isinstance(level, int):
        level = LoggingLevel(level)

    getLogger().setLevel(level=level.value)


def get_logging_level() -> LoggingLevel:
    level = getLogger().getEffectiveLevel()
    return LoggingLevel(level)


def configure(level: LoggingLevel = LoggingLevel.WARNING) -> None:
    info(f"Configuring the '{LOGGER_NAME}' logger")

    handler = logging.StreamHandler()
    fmt = "%(name)s[%(process)d] %(levelname)s %(message)s"
    handler.setFormatter(fmt=coloredlogs.ColoredFormatter(fmt=fmt))
    getLogger().addHandler(hdlr=handler)

    # Workaround for double logging
    getLogger().propagate = False

    set_logging_level(level=level)


def debug(msg: str = "") -> None:
    getLogger().debug(msg=msg)


def info(msg: str = "") -> None:
    getLogger().info(msg=msg)


def warning(msg: str = "") -> None:
    getLogger().warning(msg=msg)


def error(msg: str = "") -> None:
    getLogger().error(msg=msg)


def critical(msg: str = "") -> None:
    getLogger().critical(msg=msg)


def exception(msg: str = "") -> None:
    getLogger().exception(msg=msg)
