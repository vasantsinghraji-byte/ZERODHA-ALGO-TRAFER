# -*- coding: utf-8 -*-
"""
Money Utilities - Precision Financial Calculations
===================================================
Prevents floating-point precision errors in monetary calculations.

Problem:
    >>> 0.1 + 0.2
    0.30000000000000004

    In trading, these errors accumulate over thousands of trades,
    causing "vanishing pennies" or invalid tick sizes.

Solution:
    Use Python's decimal.Decimal for all monetary arithmetic.
    This module provides utilities for precise money handling.

Usage:
    from utils.money import Money, to_decimal, to_paise

    # Create money values
    price = Money("2500.50")
    quantity = 10
    total = price * quantity

    # Arithmetic is precise
    profit = Money("100.10") + Money("200.20")  # Exactly 300.30

    # Format for display
    print(f"Total: {total}")  # "2500.50"

    # Convert from float (use sparingly - at system boundaries)
    price = Money.from_float(2500.50)
"""

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, InvalidOperation
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


# Default precision for Indian markets (2 decimal places for rupees)
DEFAULT_PRECISION = Decimal("0.01")

# Tick size for NSE equity (5 paise = 0.05)
NSE_TICK_SIZE = Decimal("0.05")


class Money:
    """
    Precise monetary value using Decimal internally.

    Immutable value object for safe monetary arithmetic.
    Prevents floating-point precision errors that can cause:
    - Vanishing pennies over thousands of trades
    - Invalid tick sizes rejected by brokers
    - Incorrect P&L calculations

    Examples:
        >>> price = Money("2500.50")
        >>> total = price * 10
        >>> print(total)
        25005.00

        >>> profit = Money("100.10") + Money("200.20")
        >>> float(profit)  # Convert for APIs that need float
        300.30
    """

    __slots__ = ('_value',)

    def __init__(self, value: Union[str, int, Decimal, 'Money'] = 0):
        """
        Create a Money value.

        Args:
            value: String, int, or Decimal. Avoid float to prevent precision loss.

        Examples:
            Money("2500.50")  # Preferred - exact
            Money(2500)       # Integer - exact
            Money(Decimal("2500.50"))  # Decimal - exact
        """
        if isinstance(value, Money):
            self._value = value._value
        elif isinstance(value, Decimal):
            self._value = value
        elif isinstance(value, str):
            try:
                self._value = Decimal(value)
            except InvalidOperation:
                raise ValueError(f"Invalid money value: {value}")
        elif isinstance(value, int):
            self._value = Decimal(value)
        else:
            raise TypeError(
                f"Money requires str, int, or Decimal. Got {type(value).__name__}. "
                f"Use Money.from_float() for float conversion."
            )

    @classmethod
    def from_float(cls, value: float, precision: Decimal = DEFAULT_PRECISION) -> 'Money':
        """
        Create Money from float (use at system boundaries only).

        Floats are imprecise, so this immediately rounds to the specified
        precision to normalize the value.

        Args:
            value: Float value (e.g., from API response)
            precision: Rounding precision (default 0.01 for paise)

        Returns:
            Money with normalized precision

        Example:
            # From broker API that returns float
            price = Money.from_float(2500.499999999)  # -> Money("2500.50")
        """
        # Convert float to string with high precision, then to Decimal
        # This avoids float's repr issues
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(precision, rounding=ROUND_HALF_UP)
        return cls(rounded)

    @classmethod
    def from_paise(cls, paise: int) -> 'Money':
        """
        Create Money from paise (integer cents).

        Useful when storing money as integers in database.

        Args:
            paise: Value in paise (1/100 rupee)

        Returns:
            Money value

        Example:
            Money.from_paise(250050)  # -> Money("2500.50")
        """
        return cls(Decimal(paise) / 100)

    def to_paise(self) -> int:
        """
        Convert to paise (integer cents).

        Useful for storing in database as integer.

        Returns:
            Integer paise value
        """
        return int((self._value * 100).to_integral_value(rounding=ROUND_HALF_UP))

    def round_to_tick(self, tick_size: Decimal = NSE_TICK_SIZE) -> 'Money':
        """
        Round to valid tick size for exchange orders.

        NSE requires prices in multiples of 0.05 (5 paise).
        Sending 100.03 will be rejected; this rounds to 100.05.

        Args:
            tick_size: Minimum price increment (default 0.05 for NSE)

        Returns:
            Money rounded to valid tick

        Example:
            price = Money("2500.52")
            valid_price = price.round_to_tick()  # -> Money("2500.50")
        """
        # Round to nearest tick
        ticks = (self._value / tick_size).to_integral_value(rounding=ROUND_HALF_UP)
        return Money(ticks * tick_size)

    def quantize(self, precision: Decimal = DEFAULT_PRECISION) -> 'Money':
        """
        Round to specified precision.

        Args:
            precision: Decimal precision (e.g., Decimal("0.01"))

        Returns:
            Rounded Money value
        """
        return Money(self._value.quantize(precision, rounding=ROUND_HALF_UP))

    # Arithmetic operations
    def __add__(self, other: Union['Money', int]) -> 'Money':
        if isinstance(other, Money):
            return Money(self._value + other._value)
        elif isinstance(other, int):
            return Money(self._value + Decimal(other))
        return NotImplemented

    def __radd__(self, other: Union['Money', int]) -> 'Money':
        return self.__add__(other)

    def __sub__(self, other: Union['Money', int]) -> 'Money':
        if isinstance(other, Money):
            return Money(self._value - other._value)
        elif isinstance(other, int):
            return Money(self._value - Decimal(other))
        return NotImplemented

    def __rsub__(self, other: int) -> 'Money':
        if isinstance(other, int):
            return Money(Decimal(other) - self._value)
        return NotImplemented

    def __mul__(self, other: Union[int, float, Decimal]) -> 'Money':
        if isinstance(other, (int, Decimal)):
            return Money(self._value * Decimal(other))
        elif isinstance(other, float):
            # For quantity multiplication, accept float but convert
            return Money(self._value * Decimal(str(other)))
        return NotImplemented

    def __rmul__(self, other: Union[int, float, Decimal]) -> 'Money':
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float, Decimal, 'Money']) -> Union['Money', Decimal]:
        if isinstance(other, Money):
            # Money / Money = ratio (Decimal)
            return self._value / other._value
        elif isinstance(other, (int, Decimal)):
            return Money(self._value / Decimal(other))
        elif isinstance(other, float):
            return Money(self._value / Decimal(str(other)))
        return NotImplemented

    def __neg__(self) -> 'Money':
        return Money(-self._value)

    def __abs__(self) -> 'Money':
        return Money(abs(self._value))

    # Comparison operations
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Money):
            return self._value == other._value
        elif isinstance(other, (int, Decimal)):
            return self._value == Decimal(other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __lt__(self, other: Union['Money', int, Decimal]) -> bool:
        if isinstance(other, Money):
            return self._value < other._value
        elif isinstance(other, (int, Decimal)):
            return self._value < Decimal(other)
        return NotImplemented

    def __le__(self, other: Union['Money', int, Decimal]) -> bool:
        if isinstance(other, Money):
            return self._value <= other._value
        elif isinstance(other, (int, Decimal)):
            return self._value <= Decimal(other)
        return NotImplemented

    def __gt__(self, other: Union['Money', int, Decimal]) -> bool:
        if isinstance(other, Money):
            return self._value > other._value
        elif isinstance(other, (int, Decimal)):
            return self._value > Decimal(other)
        return NotImplemented

    def __ge__(self, other: Union['Money', int, Decimal]) -> bool:
        if isinstance(other, Money):
            return self._value >= other._value
        elif isinstance(other, (int, Decimal)):
            return self._value >= Decimal(other)
        return NotImplemented

    # Conversion
    def __float__(self) -> float:
        """Convert to float (for APIs that require float)."""
        return float(self._value)

    def __int__(self) -> int:
        """Convert to int (truncates)."""
        return int(self._value)

    def __str__(self) -> str:
        """Format as string with 2 decimal places."""
        return f"{self._value:.2f}"

    def __repr__(self) -> str:
        return f"Money('{self._value}')"

    def __hash__(self) -> int:
        return hash(self._value)

    def __bool__(self) -> bool:
        return self._value != 0

    @property
    def value(self) -> Decimal:
        """Get underlying Decimal value."""
        return self._value

    def format(self, currency: str = "Rs.", include_sign: bool = False) -> str:
        """
        Format for display with currency symbol.

        Args:
            currency: Currency symbol (default "Rs.")
            include_sign: Include +/- sign

        Returns:
            Formatted string

        Example:
            Money("2500.50").format()  # "Rs. 2,500.50"
            Money("-100").format(include_sign=True)  # "Rs. -100.00"
        """
        if include_sign and self._value > 0:
            return f"{currency} +{self._value:,.2f}"
        return f"{currency} {self._value:,.2f}"


# Convenience functions for common operations

def to_decimal(value: Union[float, str, int, Money], precision: Decimal = DEFAULT_PRECISION) -> Decimal:
    """
    Convert any value to Decimal with proper precision.

    Args:
        value: Value to convert
        precision: Rounding precision

    Returns:
        Decimal value
    """
    if isinstance(value, Money):
        return value.value
    elif isinstance(value, Decimal):
        return value.quantize(precision, rounding=ROUND_HALF_UP)
    elif isinstance(value, float):
        return Decimal(str(value)).quantize(precision, rounding=ROUND_HALF_UP)
    elif isinstance(value, (str, int)):
        return Decimal(value).quantize(precision, rounding=ROUND_HALF_UP)
    else:
        raise TypeError(f"Cannot convert {type(value).__name__} to Decimal")


def to_paise(rupees: Union[float, str, Decimal, Money]) -> int:
    """
    Convert rupees to paise (integer).

    Args:
        rupees: Value in rupees

    Returns:
        Integer paise value
    """
    if isinstance(rupees, Money):
        return rupees.to_paise()
    decimal_value = to_decimal(rupees)
    return int((decimal_value * 100).to_integral_value(rounding=ROUND_HALF_UP))


def from_paise(paise: int) -> Money:
    """
    Convert paise to Money.

    Args:
        paise: Integer paise value

    Returns:
        Money value
    """
    return Money.from_paise(paise)


def round_to_tick(price: Union[float, Decimal, Money], tick_size: Decimal = NSE_TICK_SIZE) -> Decimal:
    """
    Round price to valid tick size.

    Args:
        price: Price to round
        tick_size: Minimum price increment

    Returns:
        Rounded Decimal price
    """
    if isinstance(price, Money):
        return price.round_to_tick(tick_size).value
    decimal_price = to_decimal(price, Decimal("0.0001"))  # High precision first
    ticks = (decimal_price / tick_size).to_integral_value(rounding=ROUND_HALF_UP)
    return ticks * tick_size


def calculate_pnl(
    entry_price: Union[float, Money],
    exit_price: Union[float, Money],
    quantity: int,
    is_long: bool = True
) -> Money:
    """
    Calculate precise P&L for a trade.

    Args:
        entry_price: Entry price
        exit_price: Exit price
        quantity: Number of shares
        is_long: True for long position, False for short

    Returns:
        P&L as Money
    """
    entry = Money.from_float(entry_price) if isinstance(entry_price, float) else Money(entry_price)
    exit_ = Money.from_float(exit_price) if isinstance(exit_price, float) else Money(exit_price)

    if is_long:
        return (exit_ - entry) * quantity
    else:
        return (entry - exit_) * quantity


def calculate_average_price(
    current_qty: int,
    current_avg: Union[float, Money],
    new_qty: int,
    new_price: Union[float, Money]
) -> Money:
    """
    Calculate new average price after adding to position.

    Args:
        current_qty: Current position quantity
        current_avg: Current average price
        new_qty: New quantity being added
        new_price: Price of new shares

    Returns:
        New average price as Money
    """
    curr_avg = Money.from_float(current_avg) if isinstance(current_avg, float) else Money(current_avg)
    new_p = Money.from_float(new_price) if isinstance(new_price, float) else Money(new_price)

    total_qty = current_qty + new_qty
    if total_qty == 0:
        return Money(0)

    total_value = (curr_avg * current_qty) + (new_p * new_qty)
    return total_value / total_qty


# ============== TEST ==============

if __name__ == "__main__":
    print("=" * 50)
    print("MONEY UTILITIES - Test")
    print("=" * 50)

    # Demonstrate the floating-point problem
    print("\n--- Floating-Point Problem ---")
    float_result = 0.1 + 0.2
    print(f"Float: 0.1 + 0.2 = {float_result}")
    print(f"Expected: 0.3, Got: {float_result == 0.3}")

    # Show Money solution
    print("\n--- Money Solution ---")
    money_result = Money("0.1") + Money("0.2")
    print(f"Money: 0.1 + 0.2 = {money_result}")
    print(f"Exact: {money_result == Money('0.3')}")

    # Basic arithmetic
    print("\n--- Basic Arithmetic ---")
    price = Money("2500.50")
    quantity = 10
    total = price * quantity
    print(f"Price: {price}")
    print(f"Quantity: {quantity}")
    print(f"Total: {total}")

    # P&L calculation
    print("\n--- P&L Calculation ---")
    entry = Money("2500.50")
    exit_price = Money("2575.75")
    qty = 10
    pnl = (exit_price - entry) * qty
    print(f"Entry: {entry}")
    print(f"Exit: {exit_price}")
    print(f"Quantity: {qty}")
    print(f"P&L: {pnl}")

    # Tick rounding
    print("\n--- Tick Size Rounding ---")
    raw_price = Money("2500.52")
    valid_price = raw_price.round_to_tick()
    print(f"Raw price: {raw_price}")
    print(f"Valid tick: {valid_price}")

    # From float (API boundary)
    print("\n--- From Float (API Boundary) ---")
    api_price = 2500.499999999
    clean_price = Money.from_float(api_price)
    print(f"API float: {api_price}")
    print(f"Clean Money: {clean_price}")

    # Paise conversion
    print("\n--- Paise Conversion ---")
    rupees = Money("2500.50")
    paise = rupees.to_paise()
    back = Money.from_paise(paise)
    print(f"Rupees: {rupees}")
    print(f"Paise: {paise}")
    print(f"Back to Rupees: {back}")

    # Formatting
    print("\n--- Formatting ---")
    profit = Money("5250.75")
    loss = Money("-1200.50")
    print(f"Profit: {profit.format()}")
    print(f"Loss: {loss.format(include_sign=True)}")

    print("\n" + "=" * 50)
    print("Money utilities ready!")
    print("=" * 50)
