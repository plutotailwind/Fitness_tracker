import numpy as np
import torch
from typing import List, Optional


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_angle = float(np.dot(ba, bc) / den)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_joint_angles_xy(frame33_xyz):
    idx = lambda i: (frame33_xyz[i][0], frame33_xyz[i][1])
    angles = {}
    try:
        angles['elbow_l'] = calculate_angle(idx(11), idx(13), idx(15))
        angles['elbow_r'] = calculate_angle(idx(12), idx(14), idx(16))
        angles['knee_l'] = calculate_angle(idx(23), idx(25), idx(27))
        angles['knee_r'] = calculate_angle(idx(24), idx(26), idx(28))
    except Exception:
        angles = {k: 0.0 for k in ['elbow_l','elbow_r','knee_l','knee_r']}
    return angles


def preprocess_for_gcn(seq33xyz: List[np.ndarray]):
    arr = np.asarray(seq33xyz, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (33, 3):
        return torch.zeros(1, 2, 1, 33, dtype=torch.float32)
    xy = arr[:, :, :2]
    mid_hip = (xy[:, 23] + xy[:, 24]) / 2.0
    xy = xy - mid_hip[:, None, :]
    shoulder_dist = np.linalg.norm(xy[:, 11] - xy[:, 12], axis=-1) + 1e-6
    xy = xy / shoulder_dist[:, None, None]
    xy = np.transpose(xy, (2, 0, 1))
    return torch.tensor(xy, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def embed_sequence(gcn_model, seq33xyz: List[np.ndarray], device: str = 'cpu') -> np.ndarray:
    x = preprocess_for_gcn(seq33xyz).to(device)
    emb = gcn_model(x)
    emb = emb[0].permute(1, 0).contiguous().cpu().numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms
    return emb


def dtw_distance_cosine(A: np.ndarray, B: np.ndarray) -> float:
    Ta, Tb = len(A), len(B)
    if Ta == 0 or Tb == 0:
        return 1.0
    INF = 1e9
    D = np.full((Ta + 1, Tb + 1), INF, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        dots = (A[i-1:i] @ B.T).ravel()
        row_cost = 1.0 - np.clip(dots, -1.0, 1.0)
        for j in range(1, Tb + 1):
            c = row_cost[j - 1]
            D[i, j] = c + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    path_len = (Ta + Tb)
    return float(D[Ta, Tb] / path_len)


def dtw_similarity(A: np.ndarray, B: np.ndarray) -> float:
    dist = dtw_distance_cosine(A, B)
    alpha = 3.0
    sim = np.exp(-alpha * dist)
    return float(sim)


def dtw_distance_l1(A: np.ndarray, B: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    Ta, Tb = len(A), len(B)
    if Ta == 0 or Tb == 0:
        return 180.0
    Dmat = np.full((Ta + 1, Tb + 1), 1e9, dtype=np.float32)
    Dmat[0, 0] = 0.0
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(1, -1)
        w = w / (w.sum() + 1e-6)
    else:
        w = None
    for i in range(1, Ta + 1):
        diffs_raw = np.abs(A[i-1:i] - B)
        if w is not None:
            diffs = (diffs_raw * w).sum(axis=1)
        else:
            diffs = diffs_raw.mean(axis=1)
        for j in range(1, Tb + 1):
            c = diffs[j - 1]
            Dmat[i, j] = c + min(Dmat[i-1, j], Dmat[i, j-1], Dmat[i-1, j-1])
    path_len = (Ta + Tb)
    return float(Dmat[Ta, Tb] / path_len)


def compute_angles_for_seq(seq33: List[np.ndarray]) -> np.ndarray:
    def angle_from_indices(frame, i, j, k):
        p = (frame[i][0], frame[i][1])
        q = (frame[j][0], frame[j][1])
        r = (frame[k][0], frame[k][1])
        return calculate_angle(p, q, r)

    out = []
    for fr in seq33:
        try:
            elbow_l = angle_from_indices(fr, 11, 13, 15)
            elbow_r = angle_from_indices(fr, 12, 14, 16)
            knee_l = angle_from_indices(fr, 23, 25, 27)
            knee_r = angle_from_indices(fr, 24, 26, 28)
            shoulder_l = angle_from_indices(fr, 13, 11, 23)
            shoulder_r = angle_from_indices(fr, 14, 12, 24)
            hip_l = angle_from_indices(fr, 11, 23, 25)
            hip_r = angle_from_indices(fr, 12, 24, 26)
        except Exception:
            elbow_l = elbow_r = knee_l = knee_r = shoulder_l = shoulder_r = hip_l = hip_r = 0.0
        out.append([
            elbow_l, elbow_r,
            shoulder_l, shoulder_r,
            hip_l, hip_r,
            knee_l, knee_r,
        ])
    return np.asarray(out, dtype=np.float32)


def total_motion_amplitude(angles_TD: np.ndarray) -> float:
    if angles_TD.size == 0:
        return 0.0
    lo = np.percentile(angles_TD, 10, axis=0)
    hi = np.percentile(angles_TD, 90, axis=0)
    return float(np.maximum(hi - lo, 0.0).sum())


def masked_motion_amplitude(angles_TD: np.ndarray, mask_bool: np.ndarray) -> float:
    if angles_TD.size == 0:
        return 0.0
    if mask_bool is None or not np.any(mask_bool):
        return 0.0
    sub = angles_TD[:, mask_bool]
    lo = np.percentile(sub, 10, axis=0)
    hi = np.percentile(sub, 90, axis=0)
    return float(np.maximum(hi - lo, 0.0).sum())


def build_priority_mask(priority_tokens: Optional[List[str]], D: int) -> np.ndarray:
    mask = np.zeros((D,), dtype=bool)
    if priority_tokens is None:
        return mask
    tokens = {t.strip().lower() for t in priority_tokens if t and t.strip()}
    def set_pair(start_idx):
        mask[start_idx] = True; mask[start_idx+1] = True
    if "elbow" in tokens: set_pair(0)
    if "shoulder" in tokens: set_pair(2)
    if "hip" in tokens: set_pair(4)
    if "knee" in tokens: set_pair(6)
    name_to_idx = {
        "elbow_l":0, "elbow_r":1, "shoulder_l":2, "shoulder_r":3,
        "hip_l":4, "hip_r":5, "knee_l":6, "knee_r":7,
    }
    for t in tokens:
        if t in name_to_idx:
            mask[name_to_idx[t]] = True
    return mask


def smooth_angles(angles_TD: np.ndarray, window: int = 5) -> np.ndarray:
    if len(angles_TD) == 0 or window <= 1:
        return angles_TD
    w = int(window)
    pad = min(w - 1, max(0, len(angles_TD) - 1))
    padded = np.pad(angles_TD, ((pad, pad), (0, 0)), mode='reflect')
    kernel = np.ones((w, 1), dtype=np.float32) / float(w)
    sm = np.apply_along_axis(lambda col: np.convolve(col, kernel[:, 0], mode='valid'), 0, padded)
    start = (len(sm) - len(angles_TD)) // 2
    return sm[start:start + len(angles_TD)]


def resample_to_length(angles_TD: np.ndarray, target_len: int) -> np.ndarray:
    if len(angles_TD) == 0 or target_len <= 0:
        return angles_TD
    if len(angles_TD) == target_len:
        return angles_TD
    T, D = angles_TD.shape
    src_t = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(dst_t, src_t, angles_TD[:, d])
    return out


def calibrate_score(raw_score: float, gamma: float = 0.8) -> float:
    try:
        s = float(raw_score)
    except Exception:
        return 0.0
    s = max(0.0, min(1.0, s))
    return float(np.clip(s ** float(gamma), 0.0, 1.0))


