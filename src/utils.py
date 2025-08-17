"""
This module contains functions used in the core video_processing.py
module, useful to run the pogotrack pipeline.
"""

import cv2
import numpy as np
import pandas as pd
from trackpy import link_df


def get_difference(frame, background):

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

    temp = frame.copy()
    diff = cv2.subtract(temp, background)
    return diff

def binarize(frame, threshold = 2):

    """
    Function that converts the input frame into
    grayscale, and then applies a thresholding
    operation to return a binary black-and-white mask.

    Parameters
    ----------
        frame (np.ndarray):
            input image in BGR format.
        threshold (int), optional:
            threshold value used for binarization (default = 2).
    
    Return
    ------
        thresh (np.ndarray):
            binary black-and-white mask of the input image.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    Function that estimates the direction angle theta
    (in degrees) of a pogobot from its light intensity
    distribution within a circular region of radius R
    centered at the centroid.

    Parameters
    ----------
        frame (np.ndarray):
            binary image (thresh) containing light pixels.
        i0 (int):
            centroid row index (y-coordinate).
        j0 (int):
            centroid column index (x-coordinate).
        R (int), optional:
            radius of the circular region considered (default = 30).
    
    Return
    ------
        theta (float):
            estimated orientation angle in degrees.
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

def save_datas(df, frame, x, y, thetas):

    """
    Function that appends tracking data for one
    frame to the input dataframe.

    Parameters
    ----------
        df (pd.DataFrame):
            dataframe containing tracking results.
        frame (int):
            current frame index.
        x (list[int]):
            list of centroid x-coordinates.
        y (list[int]):
            list of centroid y-coordinates.
        thetas (list[float]):
            list of orientation angles in degrees.
    
    Return
    ------
        df (pd.DataFrame):
            updated dataframe including new rows.
    """

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
    Function that uses Trackpy to assign a
    unique particle ID to each detected Pogobot
    across frames, enabling trajectory linking.

    Parameters
    ----------
        df (pd.DataFrame):
            dataframe with frame, x, y, theta columns.
        search_range (int), optional:
            maximum distance (in pixels) to search
            for the same particle in consecutive frames
            (default = 50).
    
    Return
    ------
        df (pd.DataFrame):
            dataframe with an additional 'particle' column.
    """

    df = link_df(df, search_range = search_range)
    return df

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

def convert_datas(df, fps, pogobot_diameter_cm, pixel_diameter):

    """
    Function that converts tracking data from
    frame/pixels to time/cm using video fps
    and pogobot size for scaling.

    Parameters
    ----------
        df (pd.DataFrame):
            dataframe with columns: frame, x, y, theta, particle.
        fps (float):
            frames per second of the video.
        pogobot_diameter_cm (float):
            real-world Pogobot diameter in centimeters.
        pixel_diameter (float):
            measured Pogobot diameter in pixels.
    
    Return
    ------
        df (pd.DataFrame):
            transformed dataframe with columns:
            time (s), x (cm), y (cm), theta (deg), particle.
    """

    df = df.copy()
    df["time"] = df["frame"] / fps
    df["x"] = pixel_to_cm(df["x"], pogobot_diameter_cm, pixel_diameter)
    df["y"] = pixel_to_cm(df["y"], pogobot_diameter_cm, pixel_diameter)
    df = df[["time", "x", "y", "theta", "particle"]]
    return df
