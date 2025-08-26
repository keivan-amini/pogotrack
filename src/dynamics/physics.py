"""
Module containing functions related to the extraction 
of physically-relevant observables useful to characterize
pogobots' dynamics, starting form a .csv dataset containing 
time,x,y,theta informations.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.fft import fft, fftfreq
from scipy.optimize import least_squares


from .plotting import (
    plot_msd,
    plot_circle_fit,
)



def compute_omega_fft(df: pd.DataFrame):

    """
    Given a pogobot dataframe, get a measure
    of the angular velocity associated to the theta
    angle using a Fast Fourier Transform approach.

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas containing
            time, x, y, theta columns.

    Return
    ------
        omega_fft (float):
            angular velocity associated to theta angle
            computed through FFT.

    """

    time = df['time'].values
    theta_deg = df['theta'].values
    theta_rad = np.deg2rad(theta_deg)
    steps = len(time)
    dt = np.mean(np.diff(time))  # assume roughly constant timestep
    fft_vals = fft(theta_rad)
    freqs = fftfreq(steps, dt)
    fft_power = np.abs(fft_vals)**2
    idx_peak = np.argmax(fft_power[1:steps // 2]) + 1
    omega_fft = 2 * np.pi * freqs[idx_peak]  # rad/s
    return omega_fft


def compute_omega_noise(df: pd.DataFrame):

    """
    Given a pogobot dataframe, return a measure of
    the angular velocity related to the theta angle
    using the mean of the derivative of theta over time.

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas containing
            time, x, y, theta columns.

    Return
    ------
        omega_noise (float):
            angular velocity associated to theta angle
            computed through the average derivative of
            theta over time

    """

    time = df['time'].values

    theta_rad = np.unwrap(np.deg2rad(df['theta'].values))
    dtheta_dt = np.gradient(theta_rad, time)
    omega_noise = np.mean(dtheta_dt)
    return omega_noise


def compute_v_msd(df: pd.DataFrame,
                  max_tau_seconds: float = 2.0,
                  taus_percentage: float = 0.3,
                  pwm: int = None,
                  save_path: str = None,
                  ):
    
    """
    Estimate the linear velocity of a pogobot from its trajectory
    using the mean-squared displacement (MSD) method.

    The MSD is computed for increasing lag times τ, and the early-time
    regime is fitted as MSD ≈ v² τ². The slope of this fit gives an
    estimate of the squared velocity.

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas containing
            time, x, y, theta columns.
        max_tau_seconds (float):
            maximum lag time (in seconds) to consider for MSD
            computation. Default is 2.0.
        taus_percentage (float):
            fraction of the smallest lag times used for the 
            linear fit. Default is 0.3.
        save_path (str):
            path to save the plotted MSD curve and linear fit. Default None.
        pwm (int):
            PWM value, used only for labeling plots. Default None.

    Return
    ------
        v_msd (float):
            estimated pogobot linear velocity in cm/s.
    """

    x = df['x'].values
    y = df['y'].values
    time = df['time'].values
    dt = np.mean(np.diff(time))
    T = len(x)

    max_tau = int(max_tau_seconds / dt)
    taus = np.arange(1, min(max_tau, T))
    msd = []

    for tau in taus:
        dx = x[tau:] - x[:-tau]
        dy = y[tau:] - y[:-tau]
        displacement_sq = dx**2 + dy**2
        msd.append(np.mean(displacement_sq))

    taus_sec = taus * dt
    taus_sec_squared = taus_sec**2
    msd = np.array(msd)

    fit_range = int(taus_percentage * len(taus))
    x_fit = taus_sec_squared[:fit_range].reshape(-1, 1)
    y_fit = msd[:fit_range]

    reg = LinearRegression().fit(x_fit, y_fit)
    slope = reg.coef_[0]
    v_msd = np.sqrt(slope)  # final velocity estimate in cm/s

    if save_path:
        plot_msd(taus_sec_squared, msd, x_fit, reg, v_msd, pwm, save_path)

    return v_msd

def fit_circle(x, y):

    """
    Fit a circle to a set of 2D points using nonlinear least squares.

    Parameters
    ----------
    x, y : array-like
        Arrays of x and y coordinates of the trajectory.

    Returns
    -------
    xc, yc : float
        Estimated coordinates of the circle center.
    R : float
        Estimated circle radius.
    """

    # Initial guess for the circle center
    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):

        """
        Compute the distance of each point (x, y) to the circle center (xc, yc).

        Parameters
        ----------
        xc, yc : float
            Coordinates of the circle center.

        Returns
        -------
        Ri : np.ndarray
            Distances of each data point from the circle center.
        """

        return np.sqrt((x - xc)**2 + (y - yc)**2)

    def residuals(c):
        """
        Residual function for circle fitting.

        Parameters
        ----------
        c : array-like, shape (2,)
            Current estimate of circle center (xc, yc).

        Returns
        -------
        residuals : np.ndarray
            Difference between each point's radius and the mean radius.
            Used as objective function for least-squares optimization.
        """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    result = least_squares(residuals, x0=[x_m, y_m])
    xc, yc = result.x
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    return xc, yc, R


def compute_radius(df: pd.DataFrame,
                   T_subset: float = 0.5,
                   save_path: str = None,
                   pog_name: str = None,
                   pwm: int = None,
                   trial: int = None):
    """
    Estimate the local radius of curvature of a trajectory
    by fitting a circle over a temporal subset around the
    middle of the video.

    Parameters
    ----------
        df (pd.DataFrame):
            dataframe containing time, x, y, theta columns.
        T_subset (float):
            half-width of time interval (in seconds) around the
            video midpoint used for circle fitting.
            Default is 0.5.
        save_path (str):
            optional path to save circle fit plots. Default None.
        pog_name (str):
            pogobot name, only used for plotting. Default None.
        pwm (int):
            pwm value, only used for plotting. Default None.
        trial (int):
            trial index, only used for plotting. Default None.

    Return
    ------
        R (float):
            estimated radius of curvature in cm.
    """
    time = df['time'].values
    mid_video = time[len(time) // 2]
    t_min, t_max = mid_video - T_subset, mid_video + T_subset
    mask = (time >= t_min) & (time <= t_max)

    x_subset = df.loc[mask, 'x'].to_numpy()
    y_subset = df.loc[mask, 'y'].to_numpy()

    xc, yc, R = fit_circle(x_subset, y_subset)

    if save_path:
        plot_circle_fit(pog_name, pwm, trial, t_min, t_max,
                        x_subset, y_subset, xc, yc, R, save_path)

    return R

def compute_omega_ucm(v_msd: float, R: float) -> float:

    """
    Compute the angular velocity associated with uniform
    circular motion, assuming v = R * ω.

    Parameters
    ----------
        v_msd (float):
            linear velocity estimate (cm/s).
        R (float):
            radius of curvature (cm).

    Return
    ------
        omega_ucm (float):
            angular velocity in rad/s.
    """

    if R is None or R <= 0 or np.isnan(R):
        return np.nan
    return v_msd / R