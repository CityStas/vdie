"""Small Windows integration helpers used by live desktop control."""

from __future__ import annotations

import ctypes
import sys


def make_process_dpi_aware() -> bool:
    """Opt into per-monitor physical coordinates when running on Windows.

    UI Automation exposes bounding rectangles in physical screen coordinates,
    while a DPI-unaware process can see logical coordinates. Keeping the process
    DPI aware is essential for spatial target alignment on scaled displays.
    """
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        # Per-monitor-v2 context. Supported on modern Windows; older versions
        # simply fall back to the legacy SetProcessDPIAware call below.
        try:
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return True
        except Exception:
            pass
        try:
            shcore = ctypes.windll.shcore
            # PROCESS_PER_MONITOR_DPI_AWARE = 2
            result = int(shcore.SetProcessDpiAwareness(2))
            if result in (0, -2147024891):  # S_OK / already-context-set variants
                return True
        except Exception:
            pass
        try:
            return bool(user32.SetProcessDPIAware())
        except Exception:
            return False
    except Exception:
        return False
