"""
Module containing plotting functions aimed
at the study of the pogobots' characterization
dynamics.
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd

# LaTex rendering because it's nicer ;)
rc("text", usetex=True)
rc("font", family="serif")


def plot_msd(taus_sec_squared, msd, x_fit, reg, v_msd, pwm, save_path):

    """
    Plot the mean-squared displacement (MSD) curve against lag time squared,
    together with the linear regression fit used to estimate velocity.

    Parameters
    ----------
    taus_sec_squared : np.ndarray
        Lag times squared (τ²) in seconds².
    msd : np.ndarray
        Mean squared displacement values (cm²).
    x_fit : np.ndarray
        Subset of τ² values used for the linear regression fit.
    reg : sklearn.linear_model.LinearRegression
        Fitted regression model.
    v_msd : float
        Estimated velocity (cm/s) from the MSD slope.
    pwm : int
        PWM value associated with the trajectory, used in plot title.
    save_path : str or None
        If not None, path where the figure will be saved.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(taus_sec_squared, msd, label=r"MSD $(x,y)$ vs. $\tau^2$")
    plt.plot(
        x_fit,
        reg.predict(x_fit),
        "--",
        label=rf"Fit: $MSD \approx v^2 \cdot \tau^2$"
              rf"\\ $v \approx {v_msd:.2f}\,\mathrm{{cm/s}}$"
    )
    plt.xlabel(r"Lag time squared $\tau^2$ (s$^2$)")
    plt.ylabel(r"MSD (cm$^2$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(rf"Estimated velocity: PWM = {pwm}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.close()


def plot_circle_fit(pog_name, pwm, trial, t_min, t_max, x_data, y_data, xc, yc, R, save_path):

    """
    Plot the trajectory of a pogobot together with the fitted circle
    representing its local radius of curvature.

    Parameters
    ----------
    pog_name : str
        Pogobot identifier (e.g. 'pog_01').
    pwm : int
        PWM value of the trial.
    trial : int
        Trial index for the measurement.
    t_min, t_max : float
        Start and end times (in seconds) of the subset used for circle fitting.
    x_data, y_data : np.ndarray
        Subset of trajectory coordinates used in the fit.
    xc, yc : float
        Estimated coordinates of the circle center.
    R : float
        Estimated circle radius (cm).
    save_path : str or None
        If not None, path where the figure will be saved. If None,
        the figure will be shown interactively.
    """

    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_data, y_data, s=1)
    ax.plot(x_data, y_data, "b", label="Trajectory", lw=1)

    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = xc + R * np.cos(theta)
    y_circle = yc + R * np.sin(theta)

    plt.title(
        rf"Pogobot = {pog_name.replace('pog_', '')}, "
        rf"PWM = {pwm}, Trial = {trial}, "
        rf"$t_1 = {np.round(t_min, 2)}\,s$, "
        rf"$t_2 = {np.round(t_max, 2)}\,s$"
    )

    ax.plot(
        x_circle,
        y_circle,
        "r--",
        label=rf"Fitted Circle: $R = {np.round(R,2)}\,\mathrm{{cm}}$",
        lw=1,
        alpha=0.5,
    )

    ax.set_xlabel(r"$x$ (cm)")
    ax.set_ylabel(r"$y$ (cm)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi = 400)
    plt.close()


def plot_quantity(df: pd.DataFrame, pog_name: str,
                  name_pogs: list, save_path: str,
                  quantity: str, plot_config: dict):
    """
    Save plots regarding:

        1) angular velocity from FFT of theta
            as a function of pwm.
        2) linear velocity from MSD slope as
            a function of pwm.
        3) radius of curvature from circle fit
            trajectory as a function of pwm.
        4) angular velocity of trajectory as
            a function of pwm.
    
    Parameters
    ----------
        df (pd.DataFrame):
            df containing column 'pwm' and 'v_msd'. 
        pog_name (str):
            string (folder) containing the pogobot
            id related to the plot. 
        name_pogs (list):
            list containing all the pogobot id's in 
            the dynamics characterization. useful 
            to assign unique colors. 
        save_path (str):
            path to save the generated plot.
    """

    if quantity not in plot_config:
        raise ValueError(f"Unsupported quantity: {quantity}")

    config = plot_config[quantity]
    symbol, unit, ylim = config["symbol"], config["unit"], config["ylim"]

    color_map = plt.cm.get_cmap('viridis', len(name_pogs))
    pogobot_colors = {name_pogs[i]: color_map(i) for i in range(len(name_pogs))}
    color = pogobot_colors[pog_name]

    all_pwm = df['pwm'].unique()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["pwm"], df[quantity],
                s = 20, color = color, alpha = 0.6, marker= "X")
    if ylim:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.15, axis="x")
    plt.xlabel("PWM duty cycle")
    plt.ylabel(f"{symbol} {unit}")
    plt.xticks(all_pwm)
    plt.title(f"{symbol} vs PWM — ID: {pog_name.replace('pog_', '')}")
    plt.tight_layout()
    plt.savefig(save_path + ".pdf", dpi = 400, format="pdf")
    plt.close()
