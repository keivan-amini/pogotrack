"""
This module aims at implementing plot functions
useful to understand, debug and fine-tune pogotrack
processing parameters contained in config directory.
"""

import cv2
import numpy as np
import pandas as pd
import trackpy as tp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

def visualize_contours(frame, contours, x, y, thetas, cfg):

    """
    Draw centroids and direction arrows on a given frame
    and show cropped ROIs for each (pogobot) contour. 
    All parameters are read from cfg (YAML-loaded dict).

    Expected keys in cfg:
      - CENTROIDS_SIZE
      - ARROW_LENGTH_FRAME
      - TIP_LENGTH

    Parameters
    ----------
        frame (np.ndarray):
            frame whose contours are to be displayed.
        contours (list):
            list containing  [[i, j]] frame indexes
            depicting the extracted anulus contours
            for each pogobot.
        x (list):
            list containing x-coordinates of each
            pogobot in the video, measured in pixel [px].
        y (list):
            list containing y-coordinates of each
            pogobot in the video, measured in pixel [px].
        thetas (list):
            list containing the theta angle of each pogobot, 
            measured in degrees [°]. Here, $\theta \in$ [-180, 180].
        cfg (dict):
            dictionary containing all the processing parameters,
            contained in config/default.yaml
    
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

    # Subplot grid
    n = len(thetas)
    if n == 0:
        print("No pogobots detected, nothing to plot.")
        return
    if n == 1:
        cols, rows = 1, 1
    else:
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

    plt.figure(figsize=(15, 5 * rows))

    # Plot ROIs
    for i, contour in enumerate(contours):
        mask = np.zeros(frame.shape[:2], dtype = np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness = -1)
        roi = cv2.bitwise_and(frame_disp, frame_disp, mask = mask)

        xb, yb, w, h = cv2.boundingRect(contour)
        cropped = roi[yb:yb + h, xb:xb + w]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(cropped, cmap = "gray" if cropped.ndim == 2 else None)
        plt.title(f"Contour {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_arena(x, y, thetas, cfg):

    """
    Draw simple arena view with small red circles at (x, y)
    and orientation arrows.

    Expected keys in cfg:
      - ARENA_XLIM
      - ARENA_YLIM
      - ARENA_RADIUS
      - ARROW_LENGTH_VIS
      - HEAD_WIDTH
      - HEAD_LENGTH

    Parameters
    ----------
        x (list):
            list containing x-coordinates of each
            pogobot in the video, measured in pixel [px].
        y (list):
            list containing y-coordinates of each
            pogobot in the video, measured in pixel [px].
        thetas (list):
            list containing the theta angle of each pogobot, 
            measured in degrees [°]. Here, $\theta \in$ [-180, 180].
        cfg (dict):
            dictionary containing all the processing parameters,
            contained in config/default.yaml
    
    """

    plt.xlabel("x [px]")
    plt.ylabel("y [px]")
    plt.grid(True)
    plt.xlim(cfg["ARENA_XLIM"])
    plt.ylim(cfg["ARENA_YLIM"])

    for xi, yi, thetai in zip(x, y, np.radians(thetas)):
        circle = plt.Circle((xi, yi), cfg["ARENA_RADIUS"], color = "red", fill = True, linewidth = 2)
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


def plot_trajectories(csv_path, title, cfg, bg_path = None):

    """
    Plot trajectories from CSV (optionally on a background image).

    Expected keys in cfg:
      - ARENA_XLIM
      - ARENA_YLIM
      - POGOBOT_DIAMETER_CM
      - PIXEL_DIAMETER
      - FPS

    Parameters
    ----------
        csv_path (str):
            string containing the .csv data file resulted
            from the pogotrack pipeline. It must contains:

                | frame |  x  |  y  |
                |-------|-----|-----|
                |    0  |  78 |  60 |
                |    1  |  77 |  60 |
                |   ... | ... | ... |
            
        title (str):
            title of the generated plot.
        cfg (dict):
            dictionary containing all the processing parameters,
            contained in config/default.yaml
        bg_path (str), optional:
            path containing the background image, to superimpose
            the trajectory generated.

    """

    df = pd.read_csv(csv_path)
    df["frame"] = df["time"] * cfg["FPS"]
    df["x"] = df["x"] * (cfg["PIXEL_DIAMETER"] / cfg["POGOBOT_DIAMETER_CM"])
    df["y"] = df["y"] * (cfg["PIXEL_DIAMETER"] / cfg["POGOBOT_DIAMETER_CM"])

    _, ax = plt.subplots(figsize=(5, 4))
    plt.title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    #plt.grid(True)

    ax.set_xlim([0, cfg["ARENA_XLIM"][-1] * (cfg["POGOBOT_DIAMETER_CM"] / cfg["PIXEL_DIAMETER"])])
    ax.set_ylim([0, cfg["ARENA_YLIM"][-1] * (cfg["POGOBOT_DIAMETER_CM"] / cfg["PIXEL_DIAMETER"])])

    if bg_path:
        bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        bg = cv2.flip(bg, 0)
        tp.plot_traj(df, ax = ax, superimpose = bg)
    else:
        tp.plot_traj(df, ax=ax)

    plt.tight_layout()
    ax.invert_yaxis()
    plt.show()


def debug_frame(frame_masked, diff, thresh, contours, title = None):

    """
    Generate a compact debug view, containing frame, difference, 
    threshold and contour overlay.
    Runs only if cfg["DEBUG_PLOTS"] is True in config/default.yaml

    Parameters
    ----------
        frame_masked (np.ndarray):
            frame video with a circular black mask
            outside the experimental arena.
        diff (ndarray):
            frame video subtracted with the background.
        thresh (ndarray):
            thresholded frame.
        contours (list):
            list containing  [[i, j]] frame indexes
            depicting the extracted anulus contours
            for each pogobot.
        title (str), default = None:
            title of the debugging frame.

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


def save_image(img, save_path, cmap = None):

    """
    Save an image using matplotlib in a specified
    save path.

    Parameters
    ----------
        img (np.ndarray):
            image to save.
        save_path (str):
            path in which the image will be saved.
        cmap (str or Colormap):
            colormap name used to map scalar data to
            colors. Default is None.
    
    """

    plt.figure(figsize=(8, 8))
    if cmap:  # For grayscale images
        plt.imshow(img, cmap = cmap)
    else:  # For color images
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.savefig(save_path, dpi = 400, bbox_inches="tight")
    print(f"Image saved at: {save_path}")


def visualize_gif(csv_path, cfg, bg_path=None, show=True):

    """
    Save (and optionally display) a GIF overlay for debugging tracking output,
    with axes expressed in centimeters.

    Expected keys in cfg:
    --------------------
    From cfg (top-level):
      - FPS
      - POGOBOT_DIAMETER_CM
      - PIXEL_DIAMETER

    From plotting section (cfg['PLOTTING'] or cfg directly):
      - ARENA_XLIM (used only if bg is None; otherwise derived from arena size)
      - ARENA_YLIM (used only if bg is None; otherwise derived from arena size)
      - ARROW_LENGTH_VIS (interpreted as a length in **cm** for this function)
      - CENTROIDS_SIZE
      - TRAJECTORY_DPI

    Parameters
    ----------
    csv_path (str):
        Path to the CSV.
    cfg (dict):
        Full config dict or plotting dict; if full, 'PLOTTING' is used.
    bg_path (str), default = None
        Path to background image (video background).
    show (bool), default = None
        If True, display the resulting GIF in IPython when possible.

    """
    fps = cfg.get("FPS", None)
    pcfg = cfg.get("PLOTTING", cfg)

    df = pd.read_csv(csv_path)[["time", "x", "y", "theta", "particle"]].copy()
    df["particle"] = df["particle"].astype(int)

    gif_path = csv_path[:-4] + ".gif" if csv_path.lower().endswith(".csv") else (csv_path + ".gif")

    times = np.sort(df["time"].unique())
    particles = np.sort(df["particle"].unique())
    n = len(particles)
    pid_to_i = {pid: i for i, pid in enumerate(particles)}
    df["pi"] = df["particle"].map(pid_to_i).astype(int)

    T = len(times)
    xF = np.full((T, n), np.nan, float)
    yF = np.full((T, n), np.nan, float)
    thF_deg = np.full((T, n), np.nan, float)

    t_to_f = {t: k for k, t in enumerate(times)}
    f = df["time"].map(t_to_f).to_numpy()
    i = df["pi"].to_numpy()
    xF[f, i] = df["x"].to_numpy(float)    
    yF[f, i] = df["y"].to_numpy(float)    
    thF_deg[f, i] = df["theta"].to_numpy(float)

    arena_xlim = pcfg.get("ARENA_XLIM", [0, 1])
    arena_ylim = pcfg.get("ARENA_YLIM", [0, 1])

    bg = None
    extent = None
    if bg_path is not None:
        bg = plt.imread(bg_path)
        h_px, w_px = bg.shape[:2]

        cm_per_px = float(cfg["POGOBOT_DIAMETER_CM"]) / float(cfg["PIXEL_DIAMETER"])
        w_cm = w_px * cm_per_px
        h_cm = h_px * cm_per_px

        # top-left origin mapping for images:
        extent = [0.0, w_cm, h_cm, 0.0]
        arena_xlim = [0.0, w_cm]
        arena_ylim = [0.0, h_cm]

    save_dpi = int(pcfg["TRAJECTORY_DPI"])
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(arena_xlim)
    ax.set_ylim(arena_ylim)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    if bg is not None:
        bg = np.flipud(bg)  # vertical flip (along y-axis)
        ax.imshow(bg, origin="upper", extent=extent, zorder=0)  # pixel->cm mappin

    body_size = float(pcfg["CENTROIDS_SIZE"])
    scat = ax.scatter([], [], s=body_size, c="tab:blue", linewidths=0, zorder=3)

    # arrows
    base_arrow_len = float(pcfg["ARROW_LENGTH_VIS"])
    arrows = []
    for _ in range(n):
        a = FancyArrowPatch(
            (0, 0), (0, 0),
            arrowstyle="-|>",
            mutation_scale=6,
            color="red",
            linewidth=1.5,
            zorder=0,
        )
        a.set_visible(False)
        ax.add_patch(a)
        arrows.append(a)

    title = ax.set_title("")

    def update(k):
        x = xF[k]
        y = yF[k]
        th_deg = thF_deg[k]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(th_deg)

        # points
        x_plot = np.where(valid, x, np.nan)
        y_plot = np.where(valid, y, np.nan)
        scat.set_offsets(np.column_stack([x_plot, y_plot]))

        # arrows
        th = np.deg2rad(th_deg)
        dx = base_arrow_len * np.cos(th)
        dy = base_arrow_len * np.sin(th)

        for j in range(n):
            if not valid[j]:
                arrows[j].set_visible(False)
                continue
            arrows[j].set_visible(True)
            arrows[j].set_positions((x[j], y[j]), (x[j] + dx[j], y[j] + dy[j]))

        title.set_text(f"Inferred dynamics \n t = {times[k]:.3f} s")

        return (scat, title, *arrows)

    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    ani.save(gif_path, writer=PillowWriter(fps=fps), dpi=save_dpi)
    plt.close(fig)

    if show:
        try:
            from IPython.display import Image, display
            display(Image(filename=gif_path))
        except Exception:
            pass