"""
This module aims at implementing plot functions
useful to understand, debug and fine-tune pogotrack
processing parameters contained in config directory.
"""

import cv2
cv2.setNumThreads(1)
import numpy as np
import pandas as pd
import trackpy as tp
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

from src.utils import compute_led_positions, pixel_to_cm, measure_rgb_mean, frame_to_time

def show_image(title, img, cmap=None):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    if cmap:  # For grayscale images
        plt.imshow(img, cmap=cmap)
    else:  # For color images
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def zoom(frame, LEDS_coord, L=20):
    xc, yc = LEDS_coord   # (x, y)
    h, w = frame.shape[:2]

    if L % 2 != 0:
        L += 1
    half_L = L // 2

    # x must be limited by width, y by height
    xc = max(half_L, min(int(xc), w - half_L))
    yc = max(half_L, min(int(yc), h - half_L))

    # first index = y, second index = x
    zoomed_frame = frame[yc - half_L:yc + half_L, xc - half_L:xc + half_L]

    return zoomed_frame

def visualize_LEDs(frame, x, y, thetas, gamma, rho,
                   centroids_size=5, arrow_length_frame=30, tip_length=0.2,
                   visualize=False):
    leds = compute_led_positions(x, y, thetas, gamma=gamma, rho=rho)

    if not visualize:
        return leds.tolist()

    frame = np.ascontiguousarray(frame.copy())

    for (x0, y0, theta), (x2, y2) in zip(zip(x, y, thetas), leds):
        x0, y0 = int(round(x0)), int(round(y0))
        x2, y2 = int(round(x2)), int(round(y2))

        x1 = int(round(x0 + arrow_length_frame * np.cos(np.deg2rad(theta))))
        y1 = int(round(y0 + arrow_length_frame * np.sin(np.deg2rad(theta))))

        cv2.circle(frame, (x0, y0), centroids_size, (0, 255, 0), -1)
        cv2.arrowedLine(frame, (x0, y0), (x1, y1), (0, 255, 0), 2, tipLength=tip_length)
        cv2.arrowedLine(frame, (x0, y0), (x2, y2), (255, 0, 0), 2, tipLength=tip_length)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(frame_rgb)
    plt.title("LED positions")
    plt.axis("off")
    plt.show()

    return leds.tolist()


def save_rgb_debug_visualization(
    frame,
    rows_for_frame,
    out_dir,
    frame_idx,
    roi_size=30,
    zoom_size=80,
    contact_cols=6,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    half = roi_size // 2

    rois = []

    for row in rows_for_frame:
        bot_id = int(row["id"])
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        led_x = int(round(row["led_x"]))
        led_y = int(round(row["led_y"]))
        theta = float(row["theta"])
        digit = row["digit"]
        region = row["region"]
        dist = row["distance"]
        R = row["R"]
        G = row["G"]
        B = row["B"]

        x1 = max(0, led_x - half)
        y1 = max(0, led_y - half)
        x2 = min(w - 1, led_x + half)
        y2 = min(h - 1, led_y + half)

        cv2.circle(annotated, (x, y), 6, (0, 255, 0), -1)
        cv2.circle(annotated, (led_x, led_y), 6, (255, 0, 0), -1)
        cv2.line(annotated, (x, y), (led_x, led_y), (255, 255, 0), 2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)

        label1 = f"id={bot_id} d={digit} reg={region}"
        label2 = f"dist={dist:.1f} RGB=({R:.0f},{G:.0f},{B:.0f})"

        cv2.putText(annotated, label1, (x1, max(20, y1 - 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, label2, (x1, max(40, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)

        roi_big = cv2.resize(roi, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
        cv2.putText(roi_big, f"id={bot_id}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(roi_big, f"d={digit} {region}", (4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(roi_big, f"RGB=({R:.0f},{G:.0f},{B:.0f})", (4, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
        rois.append((bot_id, roi_big))

    cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}_annotated.png"), annotated)

    if rois:
        rois = sorted(rois, key=lambda t: t[0])
        tiles = [img for _, img in rois]
        tile_h, tile_w = tiles[0].shape[:2]
        n = len(tiles)
        cols = max(1, int(contact_cols))
        rows = math.ceil(n / cols)

        sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
        for k, tile in enumerate(tiles):
            r = k // cols
            c = k % cols
            sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile

        cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}_rois.png"), sheet)


def save_region_debug_visualization(
    frame,
    rows_for_frame,
    out_path,
    line_cfg,
    frame_height,
    coords_are_flipped=True,
):
    img = frame.copy()

    x1 = int(round(float(line_cfg["X1"])))
    y1 = int(round(float(line_cfg["Y1"])))
    x2 = int(round(float(line_cfg["X2"])))
    y2 = int(round(float(line_cfg["Y2"])))

    if coords_are_flipped:
        y1_draw = frame_height - 1 - y1
        y2_draw = frame_height - 1 - y2
    else:
        y1_draw = y1
        y2_draw = y2

    cv2.line(img, (x1, y1_draw), (x2, y2_draw), (255, 255, 255), 3)

    for row in rows_for_frame:
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        bot_id = int(row["id"])
        region = str(row["region"])
        color = (0, 0, 255) if region == "light" else (255, 0, 0)

        cv2.circle(img, (x, y), 16, color, 2)
        cv2.putText(
            img,
            f"id={bot_id} {region}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

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


def debug_frame(
    frame_masked,
    diff,
    thresh,
    contours=None,
    *,
    title=None,
    mode="classic",
    dark_bin=None,
    light_bin=None,
    circles=None,
):
    """
    Generate a compact debug view for classic or phototaxis processing.

    Parameters
    ----------
    frame_masked : np.ndarray
        Input frame after masking.
    diff : np.ndarray
        Difference image, or any intermediate image useful for display.
    thresh : np.ndarray
        Final thresholded / binary output used for detection.
    contours : list | None
        Contours for classic mode.
    title : str | None
        Optional figure title.
    mode : {"classic", "phototaxis"}
        Debug layout mode.
    dark_bin : np.ndarray | None
        Dark-region binary image for phototaxis mode.
    light_bin : np.ndarray | None
        Light-region binary image for phototaxis mode.
    circles : np.ndarray | list | None
        Hough circles for phototaxis mode, shape (N, 3).
    """

    def _to_display(img):
        if img is None:
            return None
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    if mode not in ("classic", "phototaxis"):
        raise ValueError(f"Unsupported debug mode: {mode}")

    if mode == "classic":
        fm_disp = _to_display(frame_masked)
        diff_disp = _to_display(diff)
        thr_disp = thresh

        overlay = frame_masked.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        if contours is not None and len(contours) > 0:
            cv2.drawContours(overlay, list(contours), -1, (0, 255, 0), 2)

        overlay = _to_display(overlay)

        plt.figure(figsize=(12, 9))
        if title:
            plt.suptitle(title)

        plt.subplot(2, 2, 1)
        plt.imshow(fm_disp, cmap="gray" if fm_disp.ndim == 2 else None)
        plt.title("Masked frame")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(diff_disp, cmap="gray" if diff_disp.ndim == 2 else None)
        plt.title("Difference")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(thr_disp, cmap="gray")
        plt.title("Threshold")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(overlay)
        plt.title("Contours")
        plt.axis("off")

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        plt.show()
        return

    fm_disp = _to_display(frame_masked)
    diff_disp = _to_display(diff)
    thr_disp = thresh
    dark_disp = dark_bin if dark_bin is not None else np.zeros_like(thresh)
    light_disp = light_bin if light_bin is not None else np.zeros_like(thresh)

    overlay = frame_masked.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    if circles is not None and len(circles) > 0:
        for c in circles:
            x, y, r = map(int, c)
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
            cv2.circle(overlay, (x, y), 2, (0, 255, 255), -1)

    overlay = _to_display(overlay)

    plt.figure(figsize=(15, 9))
    if title:
        plt.suptitle(title)

    plt.subplot(2, 3, 1)
    plt.imshow(fm_disp, cmap="gray" if fm_disp.ndim == 2 else None)
    plt.title("Masked frame")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(diff_disp, cmap="gray" if diff_disp.ndim == 2 else None)
    plt.title("Recombined / diff")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(thr_disp, cmap="gray")
    plt.title("Final binary")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(dark_disp, cmap="gray")
    plt.title("Dark binary")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(light_disp, cmap="gray")
    plt.title("Light binary")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title("Hough circles")
    plt.axis("off")

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


##massive shit

import math
from pathlib import Path

import numpy as np


def _side_of_line(x, y, x1, y1, x2, y2):
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def _coords_for_region_test(
    x,
    y,
    frame_height,
    region_test_on_unflipped_coords=True,
    coords_are_flipped=True,
):
    """Convert coordinates if the line is defined in unflipped image coordinates."""
    if region_test_on_unflipped_coords and coords_are_flipped and frame_height is not None:
        return float(x), float(frame_height - 1 - y)
    return float(x), float(y)


def classify_region_debug(x, y, line_cfg, frame_height=None, coords_are_flipped=True):
    """
    Classify a robot as 'light' or 'dark' only for visualization purposes.
    """
    x1 = float(line_cfg["X1"])
    y1 = float(line_cfg["Y1"])
    x2 = float(line_cfg["X2"])
    y2 = float(line_cfg["Y2"])

    region_test_on_unflipped_coords = bool(
        line_cfg.get("REGION_TEST_ON_UNFLIPPED_COORDS", True)
    )
    light_side = str(line_cfg.get("LIGHT_SIDE", "positive")).lower()

    xr, yr = _coords_for_region_test(
        x=x,
        y=y,
        frame_height=frame_height,
        region_test_on_unflipped_coords=region_test_on_unflipped_coords,
        coords_are_flipped=coords_are_flipped,
    )

    s = _side_of_line(xr, yr, x1, y1, x2, y2)

    if light_side == "positive":
        return "light" if s > 0 else "dark"
    if light_side == "negative":
        return "light" if s < 0 else "dark"

    raise ValueError("DIVIDING_LINE.LIGHT_SIDE must be 'positive' or 'negative'")


def _measure_frame_rows(
    frame,
    frame_idx,
    ids,
    x,
    y,
    thetas,
    led_positions,
    roi_size,
    fps,
    pogobot_diameter_cm,
    pixel_diameter,
    line_cfg=None,
    coords_are_flipped=True,
):
    """
    Build CSV rows and debug rows for one frame.

    Output CSV rows contain:
    frame, time, id, x, y, theta, led_x, led_y, R, G, B
    """
    time_s = frame_to_time(frame_idx, fps)
    csv_rows = []
    debug_rows = []

    for bot_id, x_px, y_px, theta_val, led_center in zip(ids, x, y, thetas, led_positions):
        x_px = float(x_px)
        y_px = float(y_px)
        theta_val = float(theta_val)
        led_x_px = float(led_center[0])
        led_y_px = float(led_center[1])

        R, G, B = measure_rgb_mean(frame, led_center, size=roi_size)

        row = {
            "frame": int(frame_idx),
            "time": float(time_s),
            "id": int(bot_id),
            "x": float(pixel_to_cm(x_px, pogobot_diameter_cm, pixel_diameter)),
            "y": float(pixel_to_cm(y_px, pogobot_diameter_cm, pixel_diameter)),
            "theta": theta_val,
            "led_x": float(pixel_to_cm(led_x_px, pogobot_diameter_cm, pixel_diameter)),
            "led_y": float(pixel_to_cm(led_y_px, pogobot_diameter_cm, pixel_diameter)),
            "R": float(R),
            "G": float(G),
            "B": float(B),
        }
        csv_rows.append(row)

        debug_row = {
            "frame": int(frame_idx),
            "id": int(bot_id),
            "time": float(time_s),
            "x_px": x_px,
            "y_px": y_px,
            "theta": theta_val,
            "led_x_px": led_x_px,
            "led_y_px": led_y_px,
            "R": float(R),
            "G": float(G),
            "B": float(B),
        }

        if line_cfg is not None:
            debug_row["region"] = classify_region_debug(
                x=x_px,
                y=y_px,
                line_cfg=line_cfg,
                frame_height=frame.shape[0],
            )

        debug_rows.append(debug_row)

    return csv_rows, debug_rows


def save_rgb_arena_debug_view(
    frame,
    rows_for_frame,
    out_path,
    roi_size=30,
    line_cfg=None,
):
    """
    Save one whole-arena debug image showing robot centers, LED positions,
    ROI squares, and optional light/dark labels.
    """
    img = frame.copy()
    h, w = img.shape[:2]
    half = int(roi_size) // 2

    if line_cfg is not None:
        draw_dividing_line(img, line_cfg)

    for row in rows_for_frame:
        bot_id = int(row["id"])
        x = int(round(row["x_px"]))
        y = int(round(row["y_px"]))
        led_x = int(round(row["led_x_px"]))
        led_y = int(round(row["led_y_px"]))
        R = row["R"]
        G = row["G"]
        B = row["B"]
        region = row.get("region")

        if region == "light":
            region_color = (0, 165, 255)
        elif region == "dark":
            region_color = (255, 100, 0)
        else:
            region_color = (180, 180, 180)

        x1 = max(0, led_x - half)
        y1 = max(0, led_y - half)
        x2 = min(w - 1, led_x + half)
        y2 = min(h - 1, led_y + half)

        cv2.circle(img, (x, y), 10, region_color, 2)
        cv2.circle(img, (led_x, led_y), 4, (255, 0, 255), -1)
        cv2.line(img, (x, y), (led_x, led_y), (0, 255, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        text_x = min(max(5, x1), w - 220)
        text_y = max(20, y1 - 20)

        cv2.putText(
            img, f"id={bot_id}", (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        if region is not None:
            cv2.putText(
                img, region, (text_x, text_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, region_color, 1, cv2.LINE_AA
            )
        cv2.putText(
            img, f"RGB=({R:.0f},{G:.0f},{B:.0f})", (text_x, text_y + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return img


def build_led_contact_sheet(
    frame,
    rows_for_frame,
    roi_size=30,
    zoom_size=96,
    cols=6,
):
    """
    Build a tiled zoom view of all LED ROIs for one frame.
    """
    if not rows_for_frame:
        return None

    tiles = []
    half = int(roi_size) // 2

    for row in sorted(rows_for_frame, key=lambda d: d["id"]):
        bot_id = int(row["id"])
        led_x = int(round(row["led_x_px"]))
        led_y = int(round(row["led_y_px"]))
        region = row.get("region", "")
        R = row["R"]
        G = row["G"]
        B = row["B"]

        x1 = max(0, led_x - half)
        y1 = max(0, led_y - half)
        x2 = min(frame.shape[1], led_x + half)
        y2 = min(frame.shape[0], led_y + half)

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            roi = np.zeros((roi_size, roi_size, 3), dtype=np.uint8)

        tile = cv2.resize(roi, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)

        cx = zoom_size // 2
        cy = zoom_size // 2
        cv2.drawMarker(
            tile,
            (cx, cy),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=1,
        )

        cv2.putText(
            tile, f"id={bot_id}", (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        if region:
            cv2.putText(
                tile, region, (5, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 165, 255) if region == "light" else (255, 100, 0),
                1, cv2.LINE_AA
            )
        cv2.putText(
            tile, f"RGB=({R:.0f},{G:.0f},{B:.0f})", (5, zoom_size - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA
        )

        tiles.append(tile)

    cols = max(1, int(cols))
    rows = math.ceil(len(tiles) / cols)
    tile_h, tile_w = tiles[0].shape[:2]

    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for k, tile in enumerate(tiles):
        r = k // cols
        c = k % cols
        sheet[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile

    return sheet


def save_led_contact_sheet(
    frame,
    rows_for_frame,
    out_path,
    roi_size=30,
    zoom_size=96,
    cols=6,
):
    """
    Save the tiled LED zoom view for one frame and return the image.
    """
    sheet = build_led_contact_sheet(
        frame=frame,
        rows_for_frame=rows_for_frame,
        roi_size=roi_size,
        zoom_size=zoom_size,
        cols=cols,
    )
    if sheet is None:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return sheet


def save_led_debug_gif(frames_bgr, out_path, duration=0.15):
    """
    Save an animated GIF from a list of BGR OpenCV frames.
    """
    if not frames_bgr:
        return

    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "imageio is required to save the LED debug GIF. Install it with: pip install imageio"
        ) from e

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    imageio.mimsave(str(out_path), frames_rgb, duration=float(duration))


    ##altra merda

def _side_of_line(x, y, x1, y1, x2, y2):
    """Signed side test for a point relative to a line."""
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def _normalize_dividing_line(line_cfg, frame_height):
    """
    Return the dividing line in the processed-frame coordinate system.

    line_cfg options:
        X1, Y1, X2, Y2
        LIGHT_SIDE: 'positive' or 'negative'
        COORDS: 'processed' or 'raw'
    """
    x1 = float(line_cfg["X1"])
    y1 = float(line_cfg["Y1"])
    x2 = float(line_cfg["X2"])
    y2 = float(line_cfg["Y2"])

    coords = str(line_cfg.get("COORDS", "processed")).lower()

    if coords == "raw":
        y1 = frame_height - 1 - y1
        y2 = frame_height - 1 - y2
    elif coords != "processed":
        raise ValueError("DIVIDING_LINE.COORDS must be 'processed' or 'raw'")

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "light_side": str(line_cfg.get("LIGHT_SIDE", "positive")).lower(),
    }


def classify_region_debug(x, y, line_cfg, frame_height):
    """
    Classify a robot as 'light' or 'dark' in processed-frame coordinates.
    """
    line = _normalize_dividing_line(line_cfg, frame_height)

    s = _side_of_line(
        float(x),
        float(y),
        line["x1"],
        line["y1"],
        line["x2"],
        line["y2"],
    )

    if line["light_side"] == "positive":
        return "light" if s > 0 else "dark"
    if line["light_side"] == "negative":
        return "light" if s < 0 else "dark"

    raise ValueError("DIVIDING_LINE.LIGHT_SIDE must be 'positive' or 'negative'")


def draw_dividing_line(img, line_cfg):
    """Draw the normalized dividing line on an image."""
    h = img.shape[0]
    line = _normalize_dividing_line(line_cfg, h)

    p1 = (int(round(line["x1"])), int(round(line["y1"])))
    p2 = (int(round(line["x2"])), int(round(line["y2"])))

    cv2.line(img, p1, p2, (255, 255, 255), 2)
    return img