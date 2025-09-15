# style_module.py
"""
style_module.py
This module provides an AI motion processing layer. 
It takes raw Mediapipe joints from oakd_sender.py and outputs 
smoothed or stylised motion data to be consumed by Blender.
"""

import numpy as np

# Buffer to store past frames (for temporal smoothing / ML input)
_history = []

def process_motion(joints: dict) -> dict:
    """
    Input:
        joints: dict {joint_name: [x, y, z]}
    Output:
        processed joints dict {joint_name: [x, y, z]} 
        after smoothing / AI enhancement
    """
    global _history

    # Flatten input for potential ML models
    flat = []
    for name in sorted(joints.keys()):
        flat.extend(joints[name])
    _history.append(flat)

    # Keep only the last 20 frames in history
    if len(_history) > 20:
        _history = _history[-20:]

    # ---- Baseline: simple exponential smoothing (EMA) ----
    smoothed = {}
    alpha = 0.25  # smoothing factor (0=slow, 1=fast)
    for k, v in joints.items():
        if len(v) == 3:
            prev = np.array(v)
            if smoothed.get(k) is None:
                smoothed[k] = prev
            else:
                smoothed[k] = alpha * prev + (1 - alpha) * smoothed[k]

    return {k: list(map(float, v)) for k, v in smoothed.items()}
