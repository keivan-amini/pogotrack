"""
This module contains functions used in the core video_processing.py
module, useful to run the pogotrack pipeline.
"""

import cv2
cv2.setNumThreads(1)
import numpy as np
import pandas as pd
from trackpy import link_df


def get_difference(frame, background, debug=False):

    """
    Function that creates a copy of the first
    input image, and executes the subtraction
    between the created copy and the second input
    image, using cv2.subtract() function.

    Parameters
    ----------
        frame (np.ndarray):
            subtrahend matrix.
        background (np.ndarray):
            minuend matrix.
    
    Return
    ------
        diff (np.ndarray):
            result of the subtraction.
    """
    temp = frame
    if debug:
        temp = frame.copy() # takes time
    diff = cv2.subtract(temp, background)
    return diff

def binarize(frame: np.ndarray, threshold: int = 2) -> np.ndarray:
    """
    Convert an image to grayscale if needed and apply a binary threshold.

    Parameters
    ----------
        frame (np.ndarray):
            Input image; accepts single-channel or BGR/BGRA arrays. 
        threshold (int), optional:
            Threshold value used for binarization (default = 2). 

    Returns
    -------
        thresh (np.ndarray):
            Binary black-and-white mask with values in {0, 255}. 
    """
    if frame.ndim == 3 and frame.shape[2] >= 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh


def find_contours(thresh, area_params, peri_params):

    """
    Function that extracts contours from a binary
    mask and filters them based on area and
    perimeter constraints.

    Parameters
    ----------
        thresh (np.ndarray):
            binary image from which contours are extracted.
        area_params (tuple[float, float]):
            minimum and maximum acceptable area values.
        peri_params (tuple[float, float]):
            minimum and maximum acceptable perimeter values.
    
    Return
    ------
        filtered (list[np.ndarray]):
            list of contours satisfying the constraints.
    """

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area_params[0] < area < area_params[1] and peri_params[0] < perimeter < peri_params[1]:
            filtered.append(c)
    return filtered

def get_position(contours):

    """
    Function that computes the centroid coordinates
    (x, y) of each contour using image moments.

    Parameters
    ----------
        contours (list[np.ndarray]):
            list of contours for which centroids
            are to be computed.
    
    Return
    ------
        x_coords (list[int]):
            list of centroid x-coordinates.
        y_coords (list[int]):
            list of centroid y-coordinates.
    """

    x_coords, y_coords = [], []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            x_coords.append(x)
            y_coords.append(y)
    return x_coords, y_coords

def get_angle(frame, i0, j0, R = 30):

    """
    Two bugs to solve at the moment:
    1) Parameter R must be in default.yaml
    2) Function does not work properly when Pogobots are emitting with the LED

    Function that estimates the direction angle theta
    (in degrees) of a pogobot from its light intensity
    distribution within a circular region of radius R
    centered at the centroid.

    Parameters
    ----------
        frame (np.ndarray):
            Binary image (thresh) containing light pixels.
        i0 (int):
            Centroid row index (y-coordinate).
        j0 (int):
            Centroid column index (x-coordinate).
        R (int), optional:
            Radius of the circular region considered (default = 30).

    Returns
    -------
        theta (float):
            Estimated orientation angle in degrees.
    """
    h, w = frame.shape
    i_min, i_max = max(0, i0 - R), min(h, i0 + R + 1)
    j_min, j_max = max(0, j0 - R), min(w, j0 + R + 1)

    patch = frame[i_min:i_max, j_min:j_max]
    ys = np.arange(i_min, i_max)[:, None]
    xs = np.arange(j_min, j_max)[None, :]
    mask = (ys - i0) ** 2 + (xs - j0) ** 2 <= R ** 2
    patch_masked = patch * mask

    s = patch_masked.sum()
    if s == 0:
        return np.nan

    yi = (patch_masked * ys).sum()
    xi = (patch_masked * xs).sum()

    theta = np.arctan2(yi / s - i0, xi / s - j0) * 180 / np.pi
    return theta

def get_all_angles(thresh, x, y):

    """
    Function that applies get_angle() to compute
    the orientation angle of each detected Pogobot.

    Parameters
    ----------
        thresh (np.ndarray):
            binary image (thresh) used for angle estimation.
        x (list[int]):
            list of centroid x-coordinates.
        y (list[int]):
            list of centroid y-coordinates.
    
    Return
    ------
        thetas (list[float]):
            list of orientation angles in degrees.
    """

    thetas = []
    for x0, y0 in zip(x,y):
        theta = get_angle(thresh, i0 = x0, j0 = y0)
        thetas.append(theta)
    return thetas

def compute_led_positions(x, y, thetas, gamma, rho):
    """
    Compute LED coordinates from robot center/orientation.
    Angles are in degrees.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    thetas = np.asarray(thetas, dtype=float)

    gammas = thetas + float(gamma)
    x_led = x + float(rho) * np.cos(np.deg2rad(gammas))
    y_led = y + float(rho) * np.sin(np.deg2rad(gammas))

    return np.column_stack([x_led, y_led])

def extract_square_roi(frame, center, size=30):
    """
    Extract a square ROI centered on (x, y).
    """
    xc, yc = center
    h, w = frame.shape[:2]

    size = int(size)
    if size % 2 != 0:
        size += 1
    half = size // 2

    xc = max(half, min(int(round(xc)), w - half))
    yc = max(half, min(int(round(yc)), h - half))

    roi = frame[yc - half:yc + half, xc - half:xc + half]
    return roi

def measure_rgb_mean(frame, center, size=30):
    """
    Return mean (R, G, B) from a square ROI around the LED center.
    """
    roi = extract_square_roi(frame, center, size=size)
    if roi.size == 0:
        return np.nan, np.nan, np.nan

    b = float(np.mean(roi[:, :, 0]))
    g = float(np.mean(roi[:, :, 1]))
    r = float(np.mean(roi[:, :, 2]))
    return r, g, b

def to_blue_channel(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[..., 0]
        if img.shape[2] >= 3:
            return img[..., 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



def create_line_region_masks(shape, x1, y1, x2, y2, dark_ref_x, dark_ref_y):
    h, w = shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    line_val = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
    ref_val = (x2 - x1) * (dark_ref_y - y1) - (y2 - y1) * (dark_ref_x - x1)
    if ref_val == 0:
        raise ValueError('Dark reference point lies on the dividing line.')
    dark_side = line_val > 0 if ref_val > 0 else line_val < 0
    light_side = ~dark_side
    return dark_side.astype(np.uint8) * 255, light_side.astype(np.uint8) * 255


def filter_small_components(binary: np.ndarray, min_area: int, connectivity: int = 8) -> np.ndarray:
    if min_area <= 0:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    sizes = stats[:, cv2.CC_STAT_AREA]

    keep = sizes[labels] >= min_area
    out = np.where(keep, binary, 0).astype(np.uint8)

    return out

def filter_small_components_in_rois(
    binary: np.ndarray,
    centers,
    half_size: int,
    min_area: int,
    connectivity: int = 8,
    merge_mode: str = "max",
) -> np.ndarray:
    """
    Apply connected-components area filtering only inside local ROIs.

    Parameters
    ----------
    binary : np.ndarray
        Binary input image.
    centers : iterable[(x, y)] | None
        Predicted robot centers.
    half_size : int
        Half side length of square ROI.
    min_area : int
        Minimum area to keep connected components.
    connectivity : int
        Connectivity for connected components.
    merge_mode : str
        "max" keeps a pixel if any ROI keeps it.

    Returns
    -------
    np.ndarray
        Filtered binary image.
    """
    if min_area <= 0:
        return binary

    if centers is None:
        return filter_small_components(binary, min_area=min_area, connectivity=connectivity)

    centers = list(centers)
    if len(centers) == 0:
        return filter_small_components(binary, min_area=min_area, connectivity=connectivity)

    h, w = binary.shape[:2]
    out = np.zeros_like(binary)

    for cx, cy in centers:
        x1 = max(0, int(cx - half_size))
        y1 = max(0, int(cy - half_size))
        x2 = min(w, int(cx + half_size + 1))
        y2 = min(h, int(cy + half_size + 1))

        roi = binary[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_filtered = filter_small_components(roi, min_area=min_area, connectivity=connectivity)

        if merge_mode == "max":
            out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], roi_filtered)
        else:
            out[y1:y2, x1:x2] = roi_filtered

    return out

def detect_hough_circles(binary, dp=1.0, min_dist=70.0, param1=10.0, param2=14.0,
                         min_radius=40, max_radius=60):
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    return circles[0].astype(np.float32)


def deduplicate_circles(circles: np.ndarray, center_tol: float = 12.0, radius_tol: float = 6.0) -> np.ndarray:
    if circles is None or len(circles) == 0:
        return np.empty((0, 3), dtype=np.int32)
    kept = []
    for c in circles:
        x, y, r = map(int, c)
        duplicate = False
        for k in kept:
            if np.hypot(x - k[0], y - k[1]) <= center_tol and abs(r - k[2]) <= radius_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append((x, y, r))
    return np.array(kept, dtype=np.int32)


def circles_to_centers(circles: np.ndarray):
    if circles is None or len(circles) == 0:
        return [], []
    x = [int(c[0]) for c in circles]
    y = [int(c[1]) for c in circles]
    return x, y


def process_phototaxis_frame(frame: np.ndarray,
                             background: np.ndarray,
                             arena_mask: np.ndarray,
                             line_cfg: dict,
                             dark_threshold: int,
                             light_threshold: int,
                             light_cc_enabled: bool = True,
                             light_min_area: int = 45,
                             light_connectivity: int = 8,
                             debug: bool = False):
    dark_side_mask, light_side_mask = create_line_region_masks(
        frame.shape,
        line_cfg['X1'], line_cfg['Y1'], line_cfg['X2'], line_cfg['Y2'],
        line_cfg['DARK_REF_X'], line_cfg['DARK_REF_Y']
    )
    dark_mask = cv2.bitwise_and(arena_mask, dark_side_mask)
    light_mask = cv2.bitwise_and(arena_mask, light_side_mask)

    dark_frame = to_blue_channel(frame)
    dark_background = to_blue_channel(background)
    dark_frame_masked = cv2.bitwise_and(dark_frame, dark_frame, mask=dark_mask)
    dark_background_masked = cv2.bitwise_and(dark_background, dark_background, mask=dark_mask)
    dark_diff = get_difference(dark_frame_masked, dark_background_masked, debug)
    dark_bin = binarize(dark_diff, threshold=dark_threshold)

    light_frame_masked = cv2.bitwise_and(frame, frame, mask=light_mask)
    light_background_masked = cv2.bitwise_and(background, background, mask=light_mask)
    light_diff = get_difference(light_frame_masked, light_background_masked, debug)
    light_bin = binarize(light_diff, threshold=light_threshold)
    if light_cc_enabled:
        light_bin = filter_small_components(light_bin, light_min_area, light_connectivity)

    out = cv2.bitwise_or(dark_bin, light_bin)
    out[arena_mask == 0] = 0
    

    return {
        'dark_mask': dark_mask,
        'light_mask': light_mask,
        'dark_diff': dark_diff,
        'light_diff': light_diff,
        'dark_bin': dark_bin,
        'light_bin': light_bin,
        'out': out,
    }


def detect_circles_with_fallback(binary: np.ndarray, n_pogo: int, hcfg: dict) -> np.ndarray:
    circles = detect_hough_circles(
        binary,
        dp=hcfg['DP'],
        min_dist=hcfg['MIN_DIST'],
        param1=hcfg['PARAM1'],
        param2=hcfg['PARAM2'],
        min_radius=hcfg['MIN_RADIUS'],
        max_radius=hcfg['MAX_RADIUS'],
    )
    circles = deduplicate_circles(circles)
    if len(circles) >= n_pogo:
        return circles

    circles_fb = detect_hough_circles(
        binary,
        dp=hcfg['DP'],
        min_dist=hcfg.get('FALLBACK_MIN_DIST', hcfg['MIN_DIST']),
        param1=hcfg['PARAM1'],
        param2=hcfg.get('FALLBACK_PARAM2', hcfg['PARAM2']),
        min_radius=hcfg.get('FALLBACK_MIN_RADIUS', hcfg['MIN_RADIUS']),
        max_radius=hcfg.get('FALLBACK_MAX_RADIUS', hcfg['MAX_RADIUS']),
    )
    circles_fb = deduplicate_circles(circles_fb)
    return circles_fb if len(circles_fb) > len(circles) else circles


def track_objects(df, search_range=50, memory=3):
    df = df.copy().reset_index(drop=True)
    df["row_id"] = df.index

    df_link = df.dropna(subset=["x", "y"]).copy()

    df_link = link_df(
        df_link,
        search_range=search_range,
        memory=memory,
        adaptive_stop=15,
        adaptive_step=0.95,
    )

    out = df.merge(
        df_link[["row_id", "particle"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    return out.drop(columns="row_id")


def pixel_to_cm(pixels, pogobot_diameter_cm, pixel_diameter):

    """
    Function that converts pixel distances
    to centimeters using the Pogobot diameter
    as the scale factor.

    Parameters
    ----------
        pixels (float or np.ndarray):
            distance(s) expressed in pixels.
        pogobot_diameter_cm (float):
            real-world Pogobot diameter in centimeters.
        pixel_diameter (float):
            measured Pogobot diameter in pixels.
    
    Return
    ------
        value (float or np.ndarray):
            distance(s) converted to centimeters.
    """

    scale_factor = pogobot_diameter_cm / pixel_diameter
    return pixels * scale_factor

def frame_to_time(frame_idx, fps):
    """Convert a frame index to time in seconds."""
    return float(frame_idx) / float(fps)

def convert_datas(df, fps, pogobot_diameter_cm, pixel_diameter):
    """
    Convert tracking data from frame/pixels to time/cm.
    """
    df = transform_pose_columns(
        df,
        fps=fps,
        pogobot_diameter_cm=pogobot_diameter_cm,
        pixel_diameter=pixel_diameter,
        x_col="x",
        y_col="y",
        frame_col="frame",
        out_x="x",
        out_y="y",
        out_time="time",
    )

    df = df[["time", "x", "y", "theta", "particle"]]
    return df
def transform_pose_columns(
    df,
    fps,
    pogobot_diameter_cm,
    pixel_diameter,
    x_col="x",
    y_col="y",
    frame_col="frame",
    out_x="x",
    out_y="y",
    out_time="time",
):
    df = df.copy()

    if frame_col in df.columns:
        df[out_time] = pd.to_numeric(df[frame_col], errors="coerce") / float(fps)

    df[out_x] = pixel_to_cm(
        pd.to_numeric(df[x_col], errors="coerce"),
        pogobot_diameter_cm,
        pixel_diameter,
    )
    df[out_y] = pixel_to_cm(
        pd.to_numeric(df[y_col], errors="coerce"),
        pogobot_diameter_cm,
        pixel_diameter,
    )

    return df
