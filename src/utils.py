import cv2
import numpy as np
import pandas as pd
from trackpy import link_df

def get_difference(frame, background):
    """
    Function that subtract the frame with the background.
    """
    temp = frame.copy()
    diff = cv2.subtract(temp, background)
    return diff

def binarize(frame, threshold = 2):
    """
    Given an input frame and a threshold, return the
    black and white associated mask.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def find_contours(thresh, area_params, peri_params):
    """
    Find contours that satisfy area and perimeter constraints.
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
    Get centroids (x, y) for a list of contours.
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
    Given a frame (thresh) matrix, centroids coordinates and a squared submatrix
    sized 2R, return the angle theta corresponding to the direction
    of the light center of mass (255). Consider only datas inside
    a circumference of radius R.
    """
    xi, yi, s = 0, 0, 0

    for i in range(i0 - R, i0 + R + 1):
        for j in range(j0 - R, j0 + R + 1):
            if (i - i0)**2 + (j - j0)**2 <= R**2:  # Only consider points within the circle
                yi += frame[i, j] * i
                xi += frame[i, j] * j
                s += frame[i, j]
                
    theta = np.arctan2(yi/s - i0, xi/s - j0) * 180 / np.pi
    return theta

def get_all_angles(frame, thresh, contours, x, y, centroids_size = 0, arrow_length_frame = 30, tip_length = 0.2):
    """
    Given two list of centroids x and y and a frame (thresh) matrix,
    operate the function get_angle() for each centroid.
    After that, optionally, visualize the results.
    """
    temp = frame.copy()
    thetas = []
    for x0, y0 in zip(x,y):
        theta = get_angle(thresh, i0 = x0, j0 = y0)
        thetas.append(theta)
    return thetas

def save_datas(df, frame, x, y, thetas):
    """Append tracking data for a frame to the dataframe."""
    if len(x) == 0 or len(y) == 0 or len(thetas) == 0:
        return df  # Nothing to add

    temp_df = pd.DataFrame({
        "frame": [frame] * len(x),
        "x": x,
        "y": y,
        "theta": thetas
    })

    if df.empty:
        df = temp_df
    else:
        df = pd.concat([df, temp_df], ignore_index=True)

    return df


def track_objects(df, search_range = 50):
    """
    Uses Trackpy to assign an ID to each detected pogobot.
    """
    df = link_df(df, search_range = search_range)
    return df

def pixel_to_cm(pixels, pogobot_diameter_cm, pixel_diameter):
    """Convert pixels to centimeters using the Pogobot diameter as a scale."""
    scale_factor = pogobot_diameter_cm / pixel_diameter
    return pixels * scale_factor

def convert_datas(df, fps, pogobot_diameter_cm, pixel_diameter):
    """
    Convert tracking data from frame/pixels to time/cm.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: frame, x, y, theta, particle
    fps : float
        Frames per second of the video.
    pogobot_diameter_cm : float
        Real-world Pogobot diameter in centimeters.
    pixel_diameter : float
        Measured Pogobot diameter in pixels.
    
    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with columns: time (s), x (cm), y (cm), theta (deg), particle
    """
    df = df.copy()
    df["time"] = df["frame"] / fps
    df["x"] = pixel_to_cm(df["x"], pogobot_diameter_cm, pixel_diameter)
    df["y"] = pixel_to_cm(df["y"], pogobot_diameter_cm, pixel_diameter)
    df = df[["time", "x", "y", "theta", "particle"]]
    return df
