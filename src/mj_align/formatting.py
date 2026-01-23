"""Output formatting utilities for consistent display of numeric values.

This module provides helper functions for formatting floating-point numbers,
percentages, and probabilities in a clean, readable format suitable for
command-line output and reports.
"""


def fmt_float(x: float, nd: int = 2) -> str:
    """Format a float with rounding, removing unnecessary trailing zeros.

    Provides clean numeric output by stripping trailing zeros and
    unnecessary decimal points from formatted numbers.

    Args:
        x: The floating-point number to format.
        nd: Number of decimal places for rounding. Default is 2.

    Returns:
        Formatted string representation of the number.

    Example:
        >>> fmt_float(3.14159, 2)
        '3.14'
        >>> fmt_float(5.00, 2)
        '5'
        >>> fmt_float(2.50, 2)
        '2.5'
    """
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_pct(x: float, nd: int = 4) -> str:
    """Format a proportion in [0,1] with rounding, trimming trailing zeros.

    Designed for displaying fractions or percentages in a compact format.

    Args:
        x: The proportion value (typically between 0 and 1).
        nd: Number of decimal places for rounding. Default is 4.

    Returns:
        Formatted string representation of the proportion.

    Example:
        >>> fmt_pct(0.5000, 4)
        '0.5'
        >>> fmt_pct(0.12345, 4)
        '0.1235'
        >>> fmt_pct(1.0, 4)
        '1'
    """
    s = f"{x:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def fmt_prob(p: float) -> str:
    """Format a probability or p-value in a compact, readable form.

    Handles special cases appropriately:
    - Very small values use scientific notation
    - NaN returns 'nan'
    - Zero returns '0'

    Args:
        p: The probability value to format.

    Returns:
        Formatted string representation suitable for display.

    Example:
        >>> fmt_prob(0.05)
        '0.05'
        >>> fmt_prob(0.00001)
        '1.000e-05'
        >>> fmt_prob(0.0)
        '0'
    """
    if p != p:  # NaN check (NaN != NaN is True)
        return "nan"
    if p == 0.0:
        return "0"
    if p < 1e-4:
        return f"{p:.3e}"
    return fmt_pct(p, 6)
