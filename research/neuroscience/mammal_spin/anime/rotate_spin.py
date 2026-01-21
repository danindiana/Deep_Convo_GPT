#!/usr/bin/env python3
import argparse
import shutil
import sys
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.animation import FuncAnimation


# -----------------------------
# Presets
# -----------------------------
PRESETS = {
    # Smallest practical exports (especially good for GIF size)
    "tiny": {
        "fps": 12,
        "dpi": 80,
        "bitrate": 900,
        "frame_step": 4,
        "interval": 80,  # mostly relevant for --show
    },
    # Default “good enough” quality/size balance
    "balanced": {
        "fps": 20,
        "dpi": 110,
        "bitrate": 1800,
        "frame_step": 2,
        "interval": 50,
    },
    # Larger, smoother, crisper
    "hq": {
        "fps": 30,
        "dpi": 150,
        "bitrate": 4000,
        "frame_step": 1,
        "interval": 33,
    },
}


def apply_preset(args: argparse.Namespace) -> None:
    """Apply preset values. Explicit CLI overrides win."""
    if not args.preset:
        return

    preset = PRESETS.get(args.preset)
    if preset is None:
        raise ValueError(f"Unknown preset: {args.preset}")

    # Only fill in values if user did NOT explicitly set them.
    # We detect "explicitly set" by comparing to argparse defaults.
    defaults = DEFAULTS

    if args.fps == defaults["fps"]:
        args.fps = preset["fps"]
    if args.dpi == defaults["dpi"]:
        args.dpi = preset["dpi"]
    if args.bitrate == defaults["bitrate"]:
        args.bitrate = preset["bitrate"]
    if args.frame_step == defaults["frame_step"]:
        args.frame_step = preset["frame_step"]
    if args.interval == defaults["interval"]:
        args.interval = preset["interval"]


# -----------------------------
# Core simulation
# -----------------------------
def simulate_high_dim_rotation(n_neurons: int = 100, n_steps: int = 200, seed: int = 42) -> np.ndarray:
    """Simulate rotation in high-dimensional space."""
    rng = np.random.default_rng(seed)

    vec = rng.standard_normal(n_neurons)
    vec /= np.linalg.norm(vec)

    subspace_dim = min(20, n_neurons)
    subspace = rng.standard_normal((n_neurons, subspace_dim))
    subspace, _ = np.linalg.qr(subspace)

    trajectory = []
    for i in range(n_steps):
        angle = 2 * np.pi * i / n_steps
        rot_subspace = np.eye(subspace_dim)

        if subspace_dim >= 2:
            c, s = np.cos(angle), np.sin(angle)
            rot_subspace[0, 0] = c
            rot_subspace[0, 1] = -s
            rot_subspace[1, 0] = s
            rot_subspace[1, 1] = c

        vec = subspace @ rot_subspace @ subspace.T @ vec
        vec /= np.linalg.norm(vec)
        trajectory.append(vec)

    return np.array(trajectory)


def pca_project_scaled(trajectory: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, float]:
    """PCA project to n_components and scale into ~[-1, 1]."""
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(trajectory)

    scale_factor = 0.9 / np.max(np.abs(proj))
    proj_scaled = proj * scale_factor

    explained_var = float(pca.explained_variance_ratio_[:n_components].sum() * 100)
    return proj_scaled, explained_var


# -----------------------------
# Figure / animation
# -----------------------------
def build_animation(
    trajectory: np.ndarray,
    pca_proj_scaled: np.ndarray,
    explained_var: float,
    frames: range,
    interval_ms: int,
) -> Tuple[plt.Figure, FuncAnimation]:
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title(f"3D PCA Projection\n({explained_var:.1f}% of variance)", fontweight="bold")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])

    cube_edges = [
        [[-1, -1, -1], [1, -1, -1]],
        [[1, -1, -1], [1, 1, -1]],
        [[1, 1, -1], [-1, 1, -1]],
        [[-1, 1, -1], [-1, -1, -1]],
        [[-1, -1, 1], [1, -1, 1]],
        [[1, -1, 1], [1, 1, 1]],
        [[1, 1, 1], [-1, 1, 1]],
        [[-1, 1, 1], [-1, -1, 1]],
        [[-1, -1, -1], [-1, -1, 1]],
        [[1, -1, -1], [1, -1, 1]],
        [[1, 1, -1], [1, 1, 1]],
        [[-1, 1, -1], [-1, 1, 1]],
    ]
    for edge in cube_edges:
        x = [edge[0][0], edge[1][0]]
        y = [edge[0][1], edge[1][1]]
        z = [edge[0][2], edge[1][2]]
        ax1.plot(x, y, z, "gray", alpha=0.3, linewidth=0.5)

    ax2 = fig.add_subplot(132)
    ax2.set_title("High-D Cosine Similarity Matrix", fontweight="bold")
    ax2.set_xlabel("Time point")
    ax2.set_ylabel("Time point")

    ax3 = fig.add_subplot(133)
    ax3.set_title(f"Activity in All {trajectory.shape[1]} Dimensions", fontweight="bold")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Neuron index")

    traj_line, = ax1.plot([], [], [], "b-", alpha=0.5, linewidth=2)
    current_point, = ax1.plot([], [], [], "ro", markersize=10, markeredgecolor="black", markeredgewidth=1)

    ax1.scatter(
        pca_proj_scaled[0, 0],
        pca_proj_scaled[0, 1],
        pca_proj_scaled[0, 2],
        color="green",
        s=50,
        label="Start",
        alpha=0.7,
    )

    phi = np.linspace(0, 2 * np.pi, 30)
    theta = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(phi), np.sin(theta))
    y_sphere = np.outer(np.sin(phi), np.sin(theta))
    z_sphere = np.outer(np.ones_like(phi), np.cos(theta))
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.05, color="gray", linewidth=0, antialiased=True)

    ax1.legend(loc="upper right")

    similarity_matrix = trajectory @ trajectory.T
    sim_img = ax2.imshow(
        similarity_matrix[:50, :50],
        cmap="viridis",
        aspect="auto",
        vmin=-1,
        vmax=1,
        extent=[0, 50, 50, 0],
    )
    plt.colorbar(sim_img, ax=ax2, label="Cosine similarity")

    activity_img = ax3.imshow(trajectory.T, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    plt.colorbar(activity_img, ax=ax3, label="Activation")

    time_line = ax3.axvline(x=0, color="yellow", linewidth=2, alpha=0.8)
    sim_time_line = ax2.axhline(y=0, color="red", linewidth=2, alpha=0.8)

    def update(frame_idx: int):
        traj_line.set_data(pca_proj_scaled[: frame_idx + 1, 0], pca_proj_scaled[: frame_idx + 1, 1])
        traj_line.set_3d_properties(pca_proj_scaled[: frame_idx + 1, 2])

        current_point.set_data([pca_proj_scaled[frame_idx, 0]], [pca_proj_scaled[frame_idx, 1]])
        current_point.set_3d_properties([pca_proj_scaled[frame_idx, 2]])

        time_line.set_xdata([frame_idx, frame_idx])
        sim_time_line.set_ydata([frame_idx, frame_idx])

        current_similarity = float(np.dot(trajectory[0], trajectory[frame_idx]))
        ax1.set_title(
            f"3D PCA ({explained_var:.1f}% variance)\nSimilarity: {current_similarity:.3f}",
            fontweight="bold",
        )
        return traj_line, current_point, time_line, sim_time_line

    ani = FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False)
    return fig, ani


# -----------------------------
# Export
# -----------------------------
def save_gif(ani: FuncAnimation, path: str, fps: int, dpi: int) -> None:
    try:
        from matplotlib.animation import PillowWriter
    except Exception as e:
        raise RuntimeError("GIF export requires Pillow. Install with: pip install pillow") from e
    ani.save(path, writer=PillowWriter(fps=fps), dpi=dpi)


def save_mp4(ani: FuncAnimation, path: str, fps: int, dpi: int, bitrate: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "MP4 export requires ffmpeg on PATH.\n"
            "Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "Conda: conda install -c conda-forge ffmpeg"
        )
    from matplotlib.animation import FFMpegWriter
    ani.save(path, writer=FFMpegWriter(fps=fps, bitrate=bitrate), dpi=dpi)


# -----------------------------
# CLI
# -----------------------------
DEFAULTS = {
    "fps": 20,
    "dpi": 110,
    "bitrate": 1800,
    "frame_step": 2,
    "interval": 50,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-dimensional rotation visualization with GIF/MP4 export.")
    p.add_argument("--gif", type=str, default=None, help="Output GIF path (e.g., rotation.gif)")
    p.add_argument("--mp4", type=str, default=None, help="Output MP4 path (e.g., rotation.mp4)")
    p.add_argument("--show", action="store_true", help="Show interactive window after generating/exporting")

    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Quality/size preset: tiny | balanced | hq (can still override flags below)")

    p.add_argument("--neurons", type=int, default=100, help="Number of dimensions (neurons)")
    p.add_argument("--steps", type=int, default=200, help="Simulation steps (time points)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    p.add_argument("--frame-step", type=int, default=DEFAULTS["frame_step"],
                   help="Stride through frames (higher = smaller file)")
    p.add_argument("--interval", type=int, default=DEFAULTS["interval"],
                   help="Frame interval in ms (for preview only)")

    p.add_argument("--fps", type=int, default=DEFAULTS["fps"], help="Frames per second for export")
    p.add_argument("--dpi", type=int, default=DEFAULTS["dpi"], help="DPI for export (lower = smaller)")
    p.add_argument("--bitrate", type=int, default=DEFAULTS["bitrate"], help="MP4 bitrate (lower = smaller)")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        apply_preset(args)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not args.gif and not args.mp4 and not args.show:
        print("Nothing to do: provide --gif and/or --mp4 and/or --show", file=sys.stderr)
        return 2

    if args.frame_step <= 0:
        print("--frame-step must be >= 1", file=sys.stderr)
        return 2

    trajectory = simulate_high_dim_rotation(n_neurons=args.neurons, n_steps=args.steps, seed=args.seed)
    pca_proj_scaled, explained_var = pca_project_scaled(trajectory, n_components=3)

    frames = range(0, len(pca_proj_scaled), args.frame_step)

    fig, ani = build_animation(
        trajectory=trajectory,
        pca_proj_scaled=pca_proj_scaled,
        explained_var=explained_var,
        frames=frames,
        interval_ms=args.interval,
    )

    plt.tight_layout()

    try:
        if args.gif:
            save_gif(ani, args.gif, fps=args.fps, dpi=args.dpi)
            print(f"Wrote GIF: {args.gif}")

        if args.mp4:
            save_mp4(ani, args.mp4, fps=args.fps, dpi=args.dpi, bitrate=args.bitrate)
            print(f"Wrote MP4: {args.mp4}")

    except RuntimeError as e:
        print(f"Export error: {e}", file=sys.stderr)
        plt.close(fig)
        return 1

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
