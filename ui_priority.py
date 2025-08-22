import numpy as np
import tkinter as tk
from tkinter import ttk


def build_weights_from_priority(priority_tokens, priority_weight, nonpriority_weight, D):
    """
    priority_tokens: list like ["elbow","shoulder","elbow_l","knee_r", ...]
    D must be 8; order: [elbow_l, elbow_r, shoulder_l, shoulder_r, hip_l, hip_r, knee_l, knee_r]
    Returns numpy array of shape [D].
    """
    w = np.ones((D,), dtype=np.float32) * float(nonpriority_weight)
    if priority_tokens is None:
        return w
    tokens = {t.strip().lower() for t in priority_tokens if t and t.strip()}
    # generic groups
    if "elbow" in tokens:
        w[0] = w[1] = float(priority_weight)
    if "shoulder" in tokens:
        w[2] = w[3] = float(priority_weight)
    if "hip" in tokens:
        w[4] = w[5] = float(priority_weight)
    if "knee" in tokens:
        w[6] = w[7] = float(priority_weight)
    # side-specific
    if "elbow_l" in tokens: w[0] = float(priority_weight)
    if "elbow_r" in tokens: w[1] = float(priority_weight)
    if "shoulder_l" in tokens: w[2] = float(priority_weight)
    if "shoulder_r" in tokens: w[3] = float(priority_weight)
    if "hip_l" in tokens: w[4] = float(priority_weight)
    if "hip_r" in tokens: w[5] = float(priority_weight)
    if "knee_l" in tokens: w[6] = float(priority_weight)
    if "knee_r" in tokens: w[7] = float(priority_weight)
    # normalize to keep scale stable
    s = float(w.sum())
    if s > 1e-6:
        w = w / s * D
    return w


def show_priority_ui() -> list:
    """Blocking UI to select per-joint priorities. Returns list of tokens."""
    root = tk.Tk()
    root.title("Select Joint Priorities")
    root.geometry("360x360")
    try:
        root.iconify(); root.deiconify()
    except Exception:
        pass

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame, text="Choose joints to prioritize", font=("Segoe UI", 11, "bold")).pack(pady=(0,8))

    vars_map = {}
    options = [
        ("Elbow Left", "elbow_l"), ("Elbow Right", "elbow_r"),
        ("Shoulder Left", "shoulder_l"), ("Shoulder Right", "shoulder_r"),
        ("Hip Left", "hip_l"), ("Hip Right", "hip_r"),
        ("Knee Left", "knee_l"), ("Knee Right", "knee_r"),
    ]
    grid = ttk.Frame(frame)
    grid.pack()
    for i, (label, token) in enumerate(options):
        var = tk.BooleanVar(value=(token.startswith("elbow") or token.startswith("shoulder")))
        vars_map[token] = var
        cb = ttk.Checkbutton(grid, text=label, variable=var)
        cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=6, pady=4)

    selected = []
    def on_submit():
        sel = [t for t, v in vars_map.items() if v.get()]
        selected.clear(); selected.extend(sel)
        root.destroy()

    ttk.Button(frame, text="Submit", command=on_submit).pack(pady=12)
    root.mainloop()
    return selected

def show_setup_ui() -> tuple:
    """Return (selected_priority_tokens, weights_mode) where weights_mode is 'with' or 'without'."""
    root = tk.Tk()
    root.title("Workout Setup")
    root.geometry("420x420")
    try:
        root.iconify(); root.deiconify()
    except Exception:
        pass

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Choose joints to prioritize", font=("Segoe UI", 11, "bold")).pack(pady=(0,8))

    vars_map = {}
    options = [
        ("Elbow Left", "elbow_l"), ("Elbow Right", "elbow_r"),
        ("Shoulder Left", "shoulder_l"), ("Shoulder Right", "shoulder_r"),
        ("Hip Left", "hip_l"), ("Hip Right", "hip_r"),
        ("Knee Left", "knee_l"), ("Knee Right", "knee_r"),
    ]
    grid = ttk.Frame(frame)
    grid.pack()
    for i, (label, token) in enumerate(options):
        var = tk.BooleanVar(value=(token.startswith("elbow") or token.startswith("shoulder")))
        vars_map[token] = var
        cb = ttk.Checkbutton(grid, text=label, variable=var)
        cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=6, pady=4)

    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
    ttk.Label(frame, text="Weights", font=("Segoe UI", 11, "bold")).pack(pady=(0,6))
    weights_var = tk.StringVar(value="without")
    rb_frame = ttk.Frame(frame)
    rb_frame.pack()
    ttk.Radiobutton(rb_frame, text="Without weights", variable=weights_var, value="without").grid(row=0, column=0, padx=6)
    ttk.Radiobutton(rb_frame, text="With weights", variable=weights_var, value="with").grid(row=0, column=1, padx=6)

    selected = []
    weights_mode = ["without"]
    def on_submit():
        sel = [t for t, v in vars_map.items() if v.get()]
        selected.clear(); selected.extend(sel)
        weights_mode[0] = weights_var.get()
        root.destroy()

    ttk.Button(frame, text="Submit", command=on_submit).pack(pady=12)
    root.mainloop()
    return selected, weights_mode[0]


