"""2-D scalp positions for the 18 common CHB-MIT bipolar channels.

The dashboard draws a top-down head view with each bipolar derivation placed
at the midpoint of its two electrodes. Positions are in a normalised
[-1, 1] scalp coordinate system (nose up, left ear at x=-1).

Values come from the standard 10-20 layout (Jurcak et al. 2007); we average
the x/y of the two electrodes in the pair.
"""
from __future__ import annotations

import numpy as np


# Standard 10-20 unipolar positions (x right, y forward, on unit circle scalp)
_ELECTRODES = {
    "FP1": (-0.30,  0.95), "FP2": ( 0.30,  0.95),
    "F7":  (-0.80,  0.55), "F3":  (-0.40,  0.55),
    "FZ":  ( 0.00,  0.55), "F4":  ( 0.40,  0.55), "F8": ( 0.80,  0.55),
    "T7":  (-1.00,  0.00), "C3":  (-0.50,  0.00),
    "CZ":  ( 0.00,  0.00), "C4":  ( 0.50,  0.00), "T8": ( 1.00,  0.00),
    "P7":  (-0.80, -0.55), "P3":  (-0.40, -0.55),
    "PZ":  ( 0.00, -0.55), "P4":  ( 0.40, -0.55), "P8": ( 0.80, -0.55),
    "O1":  (-0.30, -0.95), "O2":  ( 0.30, -0.95),
}


def bipolar_positions(channels: list[str]) -> np.ndarray:
    """Return (n_channels, 2) positions for a list of bipolar pair labels."""
    out = np.zeros((len(channels), 2), dtype=np.float32)
    for i, name in enumerate(channels):
        a, b = name.split("-")
        a = a.upper(); b = b.upper()
        if a not in _ELECTRODES or b not in _ELECTRODES:
            raise KeyError(f"unknown electrode in {name!r}")
        xa, ya = _ELECTRODES[a]
        xb, yb = _ELECTRODES[b]
        out[i] = (0.5 * (xa + xb), 0.5 * (ya + yb))
    return out


def head_outline() -> dict[str, np.ndarray]:
    """Points that trace the head outline + nose + ears for a top-down view."""
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    nose = np.array([[-0.12, 0.99], [0.0, 1.15], [0.12, 0.99]])
    left_ear = np.array([[-1.0, 0.12], [-1.08, 0.08], [-1.12, 0.00],
                          [-1.08, -0.08], [-1.0, -0.12]])
    right_ear = np.array([[1.0, 0.12], [1.08, 0.08], [1.12, 0.00],
                           [1.08, -0.08], [1.0, -0.12]])
    return dict(circle=circle, nose=nose, left_ear=left_ear, right_ear=right_ear)
