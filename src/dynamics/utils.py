"""
Utility modules containing functions mainly related to
cleaning pogobot datasets already generated.
"""

import pandas as pd
import numpy as np


def should_discard_trajectory(df: pd.DataFrame,
                              max_fraction: float = 0.3,
                              max_skipped: int = 4):

    """
    Check if a trajectory file should be discarded based on skipped frames.
    A skipped frame is defined as a row where x, y, theta = 0, 0, 0.
    
    Rules
    -----
    - Ignore leading and trailing skipped runs (these will be trimmed).
    - If more than `max_fraction` of *middle* rows are skipped → discard.
    - If there is a run of > `max_skipped` consecutive skipped rows in the
      *middle* → discard.
    - If every row is skipped → discard.
    
    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas
        max_fraction (float), optional:
            maximum fraction of skipped frames tolerated
            before discarding the entire dataset. Deafult is 0.3.
        max_skipped (int), optional:
            maximum allowed number of consecutive skipped frames in the
            middle of the trajectory. Default is 4.
    
    Return
    ------
        cleaned_df (pd.DataFrame or None):
            original dataframe if valid, None if it should be discarded.

    """

    if df.empty:
        return None

    skipped = (df["x"] == 0) & (df["y"] == 0) & (df["theta"] == 0)
    not_skipped = ~skipped

    # All skipped → discard immediately
    if not not_skipped.any():
        return None

    # Identify middle segment (between first and last valid rows).
    first_valid = int(np.argmax(not_skipped.values))  # first True
    last_valid = len(df) - 1 - int(np.argmax(not_skipped.values[::-1]))  # last True

    # If somehow indices cross, nothing to evaluate → discard
    if first_valid > last_valid:
        return None

    middle = skipped.iloc[first_valid : last_valid + 1]
    if middle.empty:
        return None

    # Rule 1: too many skipped frames overall (in the middle)
    if middle.mean() > max_fraction:
        return None

    # Rule 2: too many consecutive skipped frames in the middle
    consec = 0
    for is_skipped in middle:
        if is_skipped:
            consec += 1
            if consec > max_skipped:
                return None
        else:
            consec = 0

    return df


def trim_leading_trailing(df: pd.DataFrame):

    """
    Remove skipped frames (x=y=theta=0) at the beginning and at the end
    of the trajectory, preserving the middle segment.

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas
    
    Return
    ------
        cleaned_df (pd.DataFrame):
            dataframe with leading/trailing skipped frames removed.
    """

    if df.empty:
        return df

    skipped_mask = (df["x"] == 0) & (df["y"] == 0) & (df["theta"] == 0)

    # find first and last valid rows
    valid_indices = df.index[~skipped_mask]
    if valid_indices.empty:
        return pd.DataFrame(columns=df.columns)  # all skipped

    start, end = valid_indices[0], valid_indices[-1]
    return df.loc[start:end].reset_index(drop=True)


def interpolate_missing(df: pd.DataFrame):
    
    """
    Interpolate isolated skipped frames (0,0,0) between valid data.
    If more than 4 consecutive skipped frames are found, the file 
    should be discarded instead (handled externally).

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas
    
    Return
    ------
        interpolated_df (pd.DataFrame):
            dataframe with isolated skipped frames interpolated.
    """

    if df.empty:
        return df

    # Treat zeros as missing inside the middle
    mask_zero = (df["x"] == 0) & (df["y"] == 0) & (df["theta"] == 0)
    work = df.copy()

    # Replace zeros with NaN only where all three are zero
    work.loc[mask_zero, ["x", "y", "theta"]] = np.nan

    # use time column for interpolation
    work = work.sort_values("time").reset_index(drop=True)

    # Interpolate inside only (no edge extrapolation)
    work[["x", "y", "theta"]] = (
        work[["x", "y", "theta"]]
        .interpolate(method="linear", limit_area="inside")
        .bfill()  # for rare single NaNs remaining after trim; no-op if none
        .ffill()
    )

    return work


def filter_wall_hit(df: pd.DataFrame, cx, cy, radius,
                    pogobot_diameter_cm, pixel_diameter):

    """
    Given a df containing variables of a pogobot
    experiment, the center x and y coordinate (px) of the
    the experimental arena and his radius (px), filter
    outer trajectories cutting-off all datas onwards
    since a pogobot hits the wall.

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas
        cx (int):
            pixel x-coordinate of the center
            of the arena
        cy (int):
            pixel y-coordinate of the center
            of the arena
        radius (int):
            distance radius between the center
            and the wall of the arena (px)
        pogobot_diameter_cm (float):
            pogobot diameter expressed in (cm),
            useful for scaling purposes.
        pixel_diameter (float):
            pogobot diameter expressed in (px),
            useful for scaling purposes.
    
    Return
    ------
        cleaned_df (pd.DataFrame):
            dataframe with outer trajectories cleaned.
    """

    x_px = cm_to_pixel(df["x"], pogobot_diameter_cm, pixel_diameter)
    y_px = cm_to_pixel(df["y"], pogobot_diameter_cm, pixel_diameter)

    distances = np.sqrt((x_px - cx) ** 2 + (y_px - cy) ** 2)
    outside_indices = np.where(distances > radius)[0] # index where pogo hits wall
    if len(outside_indices) > 0:
        cutoff_index = outside_indices[0]
        df = df.iloc[:cutoff_index]
    return df


def cm_to_pixel(pixels, pogobot_diameter_cm, pixel_diameter):

    """
    Function that converts distances measured in cm
    to pixels using the Pogobot diameter as the scale factor.

    Parameters
    ----------
        cm (float or np.ndarray):
            distance(s) expressed in cm.
        pogobot_diameter_cm (float):
            real-world Pogobot diameter in centimeters.
        pixel_diameter (float):
            measured Pogobot diameter in pixels.
    
    Return
    ------
        value (float or np.ndarray):
            distance(s) converted to pxiels.
    """

    scale_factor = pixel_diameter / pogobot_diameter_cm
    return pixels * scale_factor
