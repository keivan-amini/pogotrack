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
from scipy.signal import butter, filtfilt
from scipy.optimize import least_squares


from .plotting import (
    plot_circle_fit,
)


### --- Signal Processing functions --- ###

def butter_lowpass(cutoff: float, fs: float, order: int = 4):

    """
    Design a low-pass Butterworth filter.

    Parameters
    ----------
        cutoff (float):
            cutoff frequency of the filter in Hz.
        fs (float):
            sampling frequency in Hz.
        order (int):
            filter order, higher values mean sharper cutoff. Default is 4.

    Return
    ------
        b, a (ndarray):
            filter coefficients.
    """

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype= 'low', analog=False)
    return b, a


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4):

    """
    Apply a low-pass Butterworth filter to an input signal.

    Parameters
    ----------
        data (ndarray):
            input signal to be filtered.
        cutoff (float):
            cutoff frequency in Hz.
        fs (float):
            sampling frequency in Hz.
        order (int):
            filter order. Default is 4.

    Return
    ------
        y (ndarray):
            filtered signal.
    """

    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y


### --- Physics observables functions --- ###

def compute_omega_fft(df: pd.DataFrame,
                      cutoff: float = 1.0,
                      order: int = 4):
    """
    Given a pogobot dataframe, get a measure of the angular velocity
    associated to the theta angle using a Fast Fourier Transform approach.
    A low-pass Butterworth filter is applied before FFT to reduce noise.

    Parameters
    ----------
        df (pd.DataFrame):
            loaded .csv dataframe in pandas containing
            time, x, y, theta columns.
        cutoff (float):
            cutoff frequency of the low-pass filter in Hz. Default is 1.0.
        order (int):
            order of the Butterworth filter. Default is 4.

    Return
    ------
        omega_fft (float):
            angular velocity associated to theta angle
            computed through FFT, in rad/s.
    """

    time = df['time'].values
    theta_deg = df['theta'].values
    theta_rad = np.deg2rad(theta_deg)
    dt = np.mean(np.diff(time))
    fs = 1.0 / dt
    steps = len(time)
    theta_rad_filt = butter_lowpass_filter(theta_rad, cutoff, fs, order)
    fft_vals = fft(theta_rad_filt)
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
    omega_noise = np.abs(np.mean(dtheta_dt))
    return omega_noise


def compute_v_msd(df: pd.DataFrame,
                  max_tau_seconds: float = 2.0,
                  taus_percentage: float = 0.3):
    """
    Estimate the linear velocity of a pogobot from its trajectory
    using the mean-squared displacement (MSD) method.

    The MSD is computed for increasing lag times τ, and the early-time
    regime is fitted as MSD ≈ v² τ². The slope of this fit gives an
    estimate of the squared velocity.

    Parameters
    ----------
        df (pd.DataFrame):
            Loaded .csv dataframe in pandas containing
            time, x, y, theta columns.
        max_tau_seconds (float):
            Maximum lag time (in seconds) to consider for MSD
            computation. Default is 2.0.
        taus_percentage (float):
            Fraction of the smallest lag times used for the 
            linear fit. Default is 0.3.

    Return
    ------
        dict:
            A dictionary containing:
            - v_msd (float): estimated pogobot linear velocity in cm/s.
            - taus_sec_squared (np.ndarray): τ² values (s²).
            - msd (np.ndarray): mean squared displacement (cm²).
            - x_fit (np.ndarray): τ² values used for the regression fit.
            - reg (LinearRegression): fitted regression model.
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

    if slope < 0:
        print(f"⚠️ Negative slope encountered (v_msd invalid), setting to 0")
        v_msd = 0
    else:
        v_msd = np.sqrt(slope)

    return {
        "v_msd": v_msd,
        "taus_sec_squared": taus_sec_squared,
        "msd": msd,
        "x_fit": x_fit,
        "reg": reg,
    }

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

    #if save_path:
        #plot_circle_fit(pog_name, pwm, trial, t_min, t_max,
                        #x_subset, y_subset, xc, yc, R, save_path) NON SALVARE AL MOMENTO I PLOT RADIUS SINGOLI

    trial_dict = {
        "trial": trial,
        "t_min": t_min,
        "t_max": t_max,
        "x_data": x_subset,
        "y_data": y_subset,
        "xc": xc,
        "yc": yc,
        "R": R
    }

    return trial_dict

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