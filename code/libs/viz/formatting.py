"""
Numbers as written in the figures (one place for every rule): values next to bars, heatmap cells, tick labels, and
the "nice" axis ends of data-scaled panels. Signed values use the true minus sign (−).
"""

import numpy as np

from viz import constants as viscons

MINUS = '−'


def _minus(text: str) -> str:
    return text.replace('-', MINUS)


def format_value(value: float, decimals: int = 2, signed: bool = False) -> str:
    """Bar values: rounded; tiny non-zero values as '<0.01' ('±<0.01' when signed: deviations get an explicit sign)."""
    if value is None or np.isnan(value):
        return '–'
    tiny = 10 ** -decimals
    if 0 < abs(value) < tiny:
        return f"{('+' if value > 0 else MINUS) if signed else ''}<{tiny:.{decimals}f}"
    return _minus(f"{value:+.{decimals}f}") if signed else f"{value:.{decimals}f}"


def cell(value: float) -> str:
    """Heatmap cell / residual box: 2 decimals, '–' without a value."""
    return f"{value:.2f}" if value is not None and np.isfinite(value) else '–'


def signed_cell(value: float) -> str:
    """Signed heatmap cell (interaction cells): '+0.12' / '−0.03', '0.00' when it rounds to zero."""
    if value is None or not np.isfinite(value):
        return '–'
    return '0.00' if abs(value) < viscons.HEATMAP_ROUNDS_TO_ZERO else _minus(f"{value:+.2f}")


def short_signed(value: float) -> str:
    """Narrow signed cell (RQ2 heatmap): '+.03', '−.12', '+1.20' (no leading zero), '0' when it rounds to zero."""
    if not np.isfinite(value):
        return '–'
    text = f"{value:+.2f}"
    if abs(value) < 1:
        text = text.replace('0.', '.', 1)
    return '0' if text in ('+.00', '-.00') else _minus(text)


def tick(value: float) -> str:
    """Compact tick label: no trailing zeros, true minus sign (e.g. −0.05, 0.005)."""
    return _minus(f"{value:.4f}".rstrip('0').rstrip('.'))


def nice_limit(extent: float, step: float = None) -> float:
    """Axis end just above the data (5 % room). step: the next multiple of it (metric bars: 0.05, at least one step);
    None: 1, 2, 2.5 or 5 x a power of ten (RQ2 forest; e.g. 0.004 -> 0.005, 0.07 -> 0.1; at least 0.01)."""
    if extent is None or not np.isfinite(extent) or extent <= 0:
        return step if step else 0.01
    if step:
        return float(np.ceil(extent * 1.05 / step) * step)
    power = 10 ** np.floor(np.log10(extent * 1.05))
    for multiple in (1, 2, 2.5, 5, 10):
        if multiple * power >= extent * 1.05:
            return float(multiple * power)
    return float(10 * power)
