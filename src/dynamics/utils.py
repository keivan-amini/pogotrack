"""
Utility modules containing functions mainly related to
cleaning pogobot datasets already generated.
"""

import pandas as pd
import numpy as np


def filter_trajectories(df: pd.DataFrame, cx, cy, radius):

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
        cleaned_df = df.iloc[:cutoff_index]
    return cleaned_df

