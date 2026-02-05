# -*- coding: utf-8 -*-
"""
Timezone Utilities for Indian Stock Trading
============================================

CRITICAL: Always use timezone-aware datetimes for trading applications!

The Problem (Bug #8 - "Server Time Trap"):
- datetime.now() returns server's local time without timezone info
- Cloud servers (AWS/Google/Azure) typically use UTC
- NSE market operates in IST (UTC+5:30)
- Using naive datetimes causes orders to fire at wrong times

Example of the bug:
- Server time: 3:45 AM UTC
- IST time: 9:15 AM IST (market open)
- datetime.now() returns 3:45 AM - your "9:15 AM" check fails!

Solution:
- Always use now_ist() instead of datetime.now()
- Use is_market_open() to check trading hours
- Use to_ist() to convert any datetime to IST

Usage:
    from utils.timezone import now_ist, is_market_open, IST

    # Get current IST time
    current_time = now_ist()

    # Check if market is open
    if is_market_open():
        execute_trade()

    # Convert UTC to IST
    ist_time = to_ist(utc_datetime)
"""

from datetime import datetime, time, timedelta
from typing import Optional
import logging

# Try to use zoneinfo (Python 3.9+) or fall back to pytz
try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo('Asia/Kolkata')
    UTC = ZoneInfo('UTC')
    _using_zoneinfo = True
except ImportError:
    try:
        from pytz import timezone as pytz_timezone
        IST = pytz_timezone('Asia/Kolkata')
        UTC = pytz_timezone('UTC')
        _using_zoneinfo = False
    except ImportError:
        raise ImportError(
            "Neither zoneinfo (Python 3.9+) nor pytz is available. "
            "Please install pytz: pip install pytz"
        )

logger = logging.getLogger(__name__)

# NSE Market Hours (IST)
MARKET_OPEN = time(9, 15)   # 9:15 AM IST
MARKET_CLOSE = time(15, 30)  # 3:30 PM IST
PRE_MARKET_OPEN = time(9, 0)  # 9:00 AM IST (pre-market starts)
POST_MARKET_CLOSE = time(16, 0)  # 4:00 PM IST (AMO orders until)


def now_ist() -> datetime:
    """
    Get current datetime in IST timezone.

    ALWAYS use this instead of datetime.now() in trading code!

    Returns:
        Timezone-aware datetime in IST

    Example:
        >>> from utils.timezone import now_ist
        >>> current = now_ist()
        >>> print(current.tzinfo)  # Asia/Kolkata
    """
    return datetime.now(IST)


def now_utc() -> datetime:
    """
    Get current datetime in UTC timezone.

    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(UTC)


def to_ist(dt: datetime) -> datetime:
    """
    Convert a datetime to IST timezone.

    Args:
        dt: A datetime object (naive or timezone-aware)

    Returns:
        Timezone-aware datetime in IST

    Note:
        - If dt is naive (no timezone), it's assumed to be UTC
        - If dt is timezone-aware, it's converted to IST
    """
    if dt.tzinfo is None:
        # Naive datetime - assume UTC
        logger.warning(
            f"Converting naive datetime {dt} to IST. "
            f"Assuming UTC. Use timezone-aware datetimes to avoid ambiguity."
        )
        if _using_zoneinfo:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = UTC.localize(dt)

    return dt.astimezone(IST)


def to_utc(dt: datetime) -> datetime:
    """
    Convert a datetime to UTC timezone.

    Args:
        dt: A datetime object (naive or timezone-aware)

    Returns:
        Timezone-aware datetime in UTC
    """
    if dt.tzinfo is None:
        # Naive datetime - assume IST (common in Indian trading apps)
        logger.warning(
            f"Converting naive datetime {dt} to UTC. "
            f"Assuming IST. Use timezone-aware datetimes to avoid ambiguity."
        )
        if _using_zoneinfo:
            dt = dt.replace(tzinfo=IST)
        else:
            dt = IST.localize(dt)

    return dt.astimezone(UTC)


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if NSE market is currently open.

    Args:
        dt: Datetime to check (default: current IST time)

    Returns:
        True if market is open, False otherwise

    Note:
        This only checks time, not holidays/weekends.
        For full market calendar, use a dedicated holiday calendar.
    """
    if dt is None:
        dt = now_ist()
    else:
        dt = to_ist(dt)

    current_time = dt.time()

    # Check if it's a weekend
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Check market hours
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def is_pre_market(dt: Optional[datetime] = None) -> bool:
    """
    Check if currently in pre-market session.

    Args:
        dt: Datetime to check (default: current IST time)

    Returns:
        True if in pre-market (9:00 AM - 9:15 AM IST)
    """
    if dt is None:
        dt = now_ist()
    else:
        dt = to_ist(dt)

    current_time = dt.time()

    if dt.weekday() >= 5:
        return False

    return PRE_MARKET_OPEN <= current_time < MARKET_OPEN


def time_until_market_open() -> Optional[timedelta]:
    """
    Get time remaining until market opens.

    Returns:
        timedelta until market open, or None if market is already open
    """
    current = now_ist()

    if is_market_open(current):
        return None

    # Create today's market open time
    market_open_today = current.replace(
        hour=MARKET_OPEN.hour,
        minute=MARKET_OPEN.minute,
        second=0,
        microsecond=0
    )

    if current.time() >= MARKET_CLOSE:
        # After market close - next open is tomorrow (or Monday)
        days_until_monday = (7 - current.weekday()) % 7 or 7
        if current.weekday() < 5:  # Weekday after close
            days_until_monday = 1
        market_open_today += timedelta(days=days_until_monday)
    elif current.weekday() >= 5:
        # Weekend - next open is Monday
        days_until_monday = (7 - current.weekday()) % 7
        market_open_today += timedelta(days=days_until_monday)

    return market_open_today - current


def market_date(dt: Optional[datetime] = None) -> datetime:
    """
    Get the trading date for a given datetime.

    If time is after market close, returns next trading day.
    If time is before market open, returns current day.

    Args:
        dt: Datetime to check (default: current IST time)

    Returns:
        Date portion as datetime at midnight IST
    """
    if dt is None:
        dt = now_ist()
    else:
        dt = to_ist(dt)

    # If after market close, next trading day
    if dt.time() > MARKET_CLOSE:
        dt = dt + timedelta(days=1)

    # Skip weekends
    while dt.weekday() >= 5:
        dt = dt + timedelta(days=1)

    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


# Convenience aliases
get_ist_now = now_ist  # Alias for compatibility
