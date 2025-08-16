import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp


def visualize_contours(frame, contours, x, y, thetas, cfg):
    """
    Draw centroids + direction arrows on the given frame and show cropped ROIs
    for each contour. All parameters are read from cfg (YAML-loaded dict).

    Expected keys in cfg:
      - CENTROIDS_SIZE
      - ARROW_LENGTH_FRAME
      - TIP_LENGTH
    """
    centroids_size = cfg["CENTROIDS_SIZE"]
    arrow_length_frame = cfg["ARROW_LENGTH_FRAME"]
    tip_length = cfg["TIP_LENGTH"]

    # Convert to RGB for Matplotlib display
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_disp = frame.copy()

    # Draw centroids + arrows
    for x0, y0, theta in zip(x, y, thetas):
        x1 = int(x0 + arrow_length_frame * np.cos(np.deg2rad(theta)))
        y1 = int(y0 + arrow_length_frame * np.sin(np.deg2rad(theta)))
        cv2.circle(frame_disp, (x0, y0), centroids_size, (0, 255, 0), -1)
        cv2.arrowedLine(frame_disp, (x0, y0), (x1, y1),
                        (0, 255, 0), 2, tipLength=tip_length)

    # --- Setup subplot grid ---
    n = len(thetas)
    if n == 0:
        print("No pogobots detected, nothing to plot.")
        return
    elif n == 1:
        cols, rows = 1, 1
    else:
        cols = min(3, n)
        rows = math.ceil(n / cols)

    plt.figure(figsize=(15, 5 * rows))

    # Plot cropped ROIs for each contour
    for i, contour in enumerate(contours):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        roi = cv2.bitwise_and(frame_disp, frame_disp, mask=mask)

        xb, yb, w, h = cv2.boundingRect(contour)
        cropped = roi[yb:yb + h, xb:xb + w]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(cropped, cmap="gray" if cropped.ndim == 2 else None)
        plt.title(f"Contour {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_arena(x, y, thetas, cfg):
    """
    Draw simple arena view with small red circles at (x, y) and orientation arrows.

    Expected keys in cfg:
      - ARENA_XLIM
      - ARENA_YLIM
      - ARENA_RADIUS
      - ARROW_LENGTH_VIS
      - HEAD_WIDTH
      - HEAD_LENGTH
    """
    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.grid(True)
    plt.xlim(cfg["ARENA_XLIM"])
    plt.ylim(cfg["ARENA_YLIM"])

    for xi, yi, thetai in zip(x, y, np.radians(thetas)):
        circle = plt.Circle((xi, yi), cfg["ARENA_RADIUS"], color="red", fill=True, linewidth=2)
        plt.gca().add_patch(circle)
        dx = cfg["ARROW_LENGTH_VIS"] * np.cos(thetai)
        dy = cfg["ARROW_LENGTH_VIS"] * np.sin(thetai)
        plt.arrow(
            xi, yi, dx, dy,
            head_width=cfg["HEAD_WIDTH"],
            head_length=cfg["HEAD_LENGTH"],
            fc="blue", ec="blue",
            length_includes_head=True
        )

    plt.show()


def plot_trajectories(csv_path, title, cfg, bg_path=None):
    """
    Plot trajectories from CSV (optionally on a background image) if cfg["PLOT_TRAJECTORIES"] is True.

    Expected keys in cfg:
      - PLOT_TRAJECTORIES (bool)
      - ARENA_XLIM
      - ARENA_YLIM
      - POGOBOT_DIAMETER_CM
      - PIXEL_DIAMETER
      - FPS
    """
    
    if cfg.get("PLOT_TRAJECTORIES", False):
        return

    df = pd.read_csv(csv_path)
    df["frame"] = df["time"] * cfg["FPS"]
    df["x"] = df["x"] * (cfg["PIXEL_DIAMETER"] / cfg["POGOBOT_DIAMETER_CM"])
    df["y"] = df["y"] * (cfg["PIXEL_DIAMETER"] / cfg["POGOBOT_DIAMETER_CM"])

    _, ax = plt.subplots(figsize=(5, 4))
    plt.title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    plt.grid(True)

    ax.set_xlim([0, cfg["ARENA_XLIM"][-1] * (cfg["POGOBOT_DIAMETER_CM"] / cfg["PIXEL_DIAMETER"])])
    ax.set_ylim([0, cfg["ARENA_YLIM"][-1] * (cfg["POGOBOT_DIAMETER_CM"] / cfg["PIXEL_DIAMETER"])])

    if bg_path:
        bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        bg = cv2.flip(bg, 0)
        tp.plot_traj(df, ax=ax, superimpose=bg)
    else:
        tp.plot_traj(df, ax=ax)

    plt.tight_layout()
    ax.invert_yaxis()
    plt.show()


def debug_frame(frame_masked, diff, thresh, contours, title=None):
    """
    Compact debug view: masked frame, difference, threshold, and contour overlay.
    This function does not gate itself; call it only when your config says to.
    """
    fm_disp = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2RGB) if frame_masked.ndim == 3 else frame_masked
    diff_disp = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB) if diff.ndim == 3 else diff
    thr_disp = thresh

    overlay = frame_masked.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, list(contours), -1, (0, 255, 0), 2)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 9))
    if title:
        plt.suptitle(title)

    plt.subplot(2, 2, 1); plt.imshow(fm_disp); plt.title("Masked frame"); plt.axis("off")
    plt.subplot(2, 2, 2); plt.imshow(diff_disp, cmap="gray" if diff_disp.ndim == 2 else None); plt.title("Difference"); plt.axis("off")
    plt.subplot(2, 2, 3); plt.imshow(thr_disp, cmap="gray"); plt.title("Threshold"); plt.axis("off")
    plt.subplot(2, 2, 4); plt.imshow(overlay); plt.title("Contours"); plt.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
