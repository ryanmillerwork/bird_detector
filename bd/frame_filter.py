from __future__ import annotations

import cv2
import numpy as np


def is_corrupted_frame(frame, low_std_thresh: float = 2.0, flat_row_ratio: float = 0.3) -> bool:
    """Heuristic to reject obviously corrupted frames."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return True

    # If the frame is effectively a single color, treat as corrupted
    if len(np.unique(frame)) <= 1:
        return True

    # Check bottom half for large runs of near-flat rows
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    bottom = gray[h // 2 :]
    row_std = bottom.std(axis=1)
    flat_rows = (row_std < low_std_thresh).sum()
    if flat_rows / max(len(row_std), 1) > flat_row_ratio:
        return True

    return False





