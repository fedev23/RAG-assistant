import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError


def get_now_iso_and_epoch(timezone_name: str, logger: logging.Logger) -> tuple[str, int]:
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning("Invalid timezone '%s'. Falling back to UTC.", timezone_name)
        tz = ZoneInfo("UTC")

    now = datetime.now(tz=tz)
    return now.isoformat(), int(now.timestamp())
