"""Timestamp arithmetic for model-selected structured schedules; no intent parsing."""
from datetime import datetime, timedelta, timezone
import math


class MissingScheduleError(ValueError):
    """A model supplied a placeholder instead of a usable reminder time."""


def resolve_execution_time(value: object, received: datetime, now: datetime) -> datetime:
    if value is None:
        return now
    if isinstance(value, dict) and set(value) == {'at'}:
        return resolve_execution_time(value['at'], received, now)
    if isinstance(value, dict) and set(value) == {'immediate'}:
        if value['immediate'] is not True:
            raise ValueError('immediate must be true')
        return now
    if isinstance(value, dict):
        if set(value) != {'delay_seconds'}:
            raise ValueError('Expected delay_seconds')
        delay = value['delay_seconds']
        if type(delay) in (int, float) and delay == 0:
            raise MissingScheduleError('Reminder time is missing: zero delay is not a schedule. Ask the user when to remind them; do not assume now.')
        if type(delay) not in (int, float) or not 0 < delay <= 365 * 86400 or not math.isfinite(delay):
            raise ValueError('Delay must be positive and at most 365 days')
        return received + timedelta(seconds=delay)
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected aware ISO timestamp, relative delay, or null')
    execution = datetime.fromisoformat(value)
    if execution.utcoffset() is None:
        raise ValueError('Execution time must include a timezone')
    if execution < received:
        raise ValueError('Execution time is in the past; use null for immediate actions')
    return execution.astimezone(timezone.utc)
