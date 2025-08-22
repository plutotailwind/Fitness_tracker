import numpy as np
from typing import List, Optional


def compute_forward_vector_3d(frame33_xyz) -> Optional[np.ndarray]:
    try:
        ls = np.array(frame33_xyz[11], dtype=np.float32)
        rs = np.array(frame33_xyz[12], dtype=np.float32)
        lh = np.array(frame33_xyz[23], dtype=np.float32)
        rh = np.array(frame33_xyz[24], dtype=np.float32)
    except Exception:
        return None
    mid_sh = 0.5 * (ls + rs)
    mid_hip = 0.5 * (lh + rh)
    shoulder_vec = rs - ls
    up_vec = mid_sh - mid_hip
    fwd = np.cross(shoulder_vec, up_vec)
    n = np.linalg.norm(fwd) + 1e-8
    if n < 1e-6:
        return None
    return (fwd / n).astype(np.float32)


def average_forward_vector(seq33_xyz: List[np.ndarray]) -> Optional[np.ndarray]:
    acc = np.zeros(3, dtype=np.float32)
    count = 0
    for fr in seq33_xyz:
        v = compute_forward_vector_3d(fr)
        if v is not None and np.isfinite(v).all():
            acc += v
            count += 1
    if count == 0:
        return None
    v = acc / (np.linalg.norm(acc) + 1e-8)
    return v.astype(np.float32)


