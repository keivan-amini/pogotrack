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
    
    If more than `max_fraction` (default 30%) of rows are skipped,
    discard the entire dataframe.
    If there are more than `max_skipped` consecutive skipped frames
    in the middle of the trajectory, discard the dataframe.
    Otherwise return the dataframe unchanged.
    
    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas
        max_fraction (float), optional:
            maximum fraction of skipped frames tolerated
            before discarding the entire dataset. Deafult is 0.3.
        max_skipped (int), optional:
            Maximum allowed number of consecutive skipped frames (except
            at the beginning or end of the trajectory). Default is 4.
    
    Return
    ------
        cleaned_df (pd.DataFrame or None):
            original dataframe if valid,
            None if discarded (too many skipped frames).

    """
    skipped_mask = (df['x'] == 0) & (df['y'] == 0) & (df['theta'] == 0)
    skipped_fraction = skipped_mask.mean()

    # Rule 1: Too many skipped frames overall
    if skipped_fraction > max_fraction:
        return None

    # Rule 2: Too many consecutive skips in the middle
    consecutive = 0
    for i, skipped in enumerate(skipped_mask):
        if skipped:
            consecutive += 1
        else:
            if consecutive > max_skipped and i - consecutive > 0 and i < len(df):
                return None
            consecutive = 0

    return df


def trim_leading_trailing(df: pd.DataFrame):

    """
    Trim skipped frames at the beginning and end of the
    dataframe, leaving only valid data in the middle.

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas
    
    Return
    ------
        cleaned_df (pd.DataFrame):
            dataframe with leading/trailing skipped frames removed.
    """

    is_skipped = (df[['x', 'y', 'theta']] == 0).all(axis = 1)
    first_valid = df.index[~is_skipped].min()
    last_valid = df.index[~is_skipped].max()
    if pd.isna(first_valid) or pd.isna(last_valid):
        # no valid rows at all
        return df.iloc[0:0]

    return df.loc[first_valid:last_valid].reset_index(drop = True)


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

    df = df.copy()
    mask = (df[['x', 'y', 'theta']] == 0).all(axis=1)
    df.loc[mask, ['x', 'y', 'theta']] = np.nan
    df[['x', 'y', 'theta']] = (
        df[['x', 'y', 'theta']]
        .interpolate(method="linear")
        .bfill()
        .ffill())
    return df


def filter_wall_hit(df: pd.DataFrame, cx, cy, radius):

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
    
    Return
    ------
        cleaned_df (pd.DataFrame):
            dataframe with outer trajectories cleaned.
    """

    distances = np.sqrt((df['x'] - cx) ** 2 + (df['y'] - cy) ** 2)
    outside_indices = np.where(distances > radius)[0] # index where pogo hits wall
    if len(outside_indices) > 0:
        cutoff_index = outside_indices[0]
        df = df.iloc[:cutoff_index]
    return df
