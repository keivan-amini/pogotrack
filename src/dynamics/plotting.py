"""
Module containing plotting functions aimed
at the study of the pogobots' characterization
dynamics.
"""

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import matplotlib.cm as cm

# LaTex rendering because it's nicer ;)
rc("text", usetex=True)
rc("font", family="serif")



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
        2) noisy angular velocity computed as
            the mean of the derivative of
            theta over time, as a function of pwm.
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


def plot_msd_trials(trials_data, pwm, save_path=None):

    """
    Plot overlay of MSD curves and linear fits for all trials
    of a given PWM.

    Parameters
    ----------
    trials_data (list of dict):
        List of dictionaries returned by compute_v_msd(), one per trial.
    pwm (int):
        PWM value, used for labeling the plot.
    save_path (str or None):
        If not None, path where the figure will be saved.
    """

    plt.figure(figsize=(4, 3))

    for i, trial_data in enumerate(trials_data):
        taus = trial_data["taus_sec_squared"]
        msd = trial_data["msd"]
        x_fit = trial_data["x_fit"]
        reg = trial_data["reg"]
        v_msd = trial_data["v_msd"]

        plt.plot(taus, msd, alpha = 0.6, label=f"Trial {i}: MSD", color = "skyblue")
        plt.plot(x_fit, reg.predict(x_fit), alpha = 0.3,
                color = "steelblue")

        #label=fr"Trial {i}: $v\approx {v_msd:.2f}\,\mathrm{{cm/s}}$"

    plt.xlabel(r"$\tau^2$ (s$^2$)")
    plt.ylabel(r"MSD (cm$^2$)")
    plt.ylim(0, 100)
    plt.title(rf"PWM = {pwm}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path + ".pdf", dpi = 400, format="pdf")
    plt.close()


### --- subplots sections --- #

def plot_msd_grid(all_trials_per_pwm: dict,
                  save_path: str = None):
    """
    Generate a single figure with MSD curves and fits for all PWM values
    of a given pogobot, arranged in a grid of subplots.

    Each subplot corresponds to one PWM and overlays MSD curves across
    all its trials, together with the linear fits used to estimate v_msd.

    Parameters
    ----------
    all_trials_per_pwm (dict):
        Dictionary mapping PWM values to lists of trial_data dicts.
        Each trial_data should come from `compute_v_msd` and contain
        {"taus_sec_squared", "msd", "x_fit", "reg", "v_msd"}.
    save_path (str):
        Path to save the combined figure. If None, the figure is not saved.
    """
    n_pwms = len(all_trials_per_pwm)
    ncols = 5
    nrows = int(np.ceil(n_pwms / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2 * ncols, 2 * nrows),
                             sharex=False, sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(hspace = 0.5, bottom=0.15)

    for ax in axes:
        ax.axis("off")
        ax.set_ylim(-10, 100)

    last_row_pwms = n_pwms % ncols if n_pwms % ncols != 0 else ncols
    shift_frac = (ncols - last_row_pwms) / (2 * ncols) - 0.014 # fraction of figure width to shift

    for i, (pwm, trials_data) in enumerate(all_trials_per_pwm.items()):
        row = i // ncols
        col = i % ncols

        ax = axes[i]
        ax.axis("on")

        # Shift axes in the last row
        if row == nrows - 1 and last_row_pwms < ncols:
            pos = ax.get_position()
            new_x0 = pos.x0 + shift_frac
            ax.set_position([new_x0, pos.y0, pos.width, pos.height])

        for j, trial_data in enumerate(trials_data):
            taus = trial_data["taus_sec_squared"]
            msd = trial_data["msd"]
            x_fit = trial_data["x_fit"]
            reg = trial_data["reg"]

            ax.plot(taus, msd, alpha=0.4, label=f"Trial {j}", color="skyblue")
            ax.plot(x_fit, reg.predict(x_fit), alpha=0.6, color="steelblue")
            ax.set_xticks([0,1,2])
            ax.set_yticks([0,50,100])

        ax.set_title(f"PWM = {pwm}")

        if row == nrows - 1:
            ax.set_xlabel(r"$\tau^2$ (s$^2$)")
        if col == 0:
            ax.set_ylabel(r"MSD (cm$^2$)")   


    if save_path:
        plt.savefig(save_path + ".pdf", dpi=400, format="pdf")
    plt.close(fig)




def plot_circle_fit_grid(all_trials_per_pwm: dict,
                         save_path: str = None):
    """
    Generate a grid figure showing trajectory + circle fit
    for each PWM value of a given pogobot.

    Each subplot corresponds to one PWM, and shows *all trials*
    (trajectories + fitted circles).

    Parameters
    ----------
    all_trials_per_pwm : dict
        Dictionary mapping PWM values -> list of trial dicts.
        Each trial dict should contain:
            {
              "trial": int,
              "t_min": float,
              "t_max": float,
              "x_data": np.ndarray,
              "y_data": np.ndarray,
              "xc": float,
              "yc": float,
              "R": float
            }
    save_path : str or None
        If provided, saves the combined figure to this path.
    """
    n_pwms = len(all_trials_per_pwm)
    ncols = 5
    nrows = int(np.ceil(n_pwms / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2 * ncols, 2 * nrows),
                             sharex=False, sharey=True)
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.5, bottom=0.15)

    # Start with all axes off
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 200)
        ax.set_ylim(-10, 200)

    # Compute centering shift for last row
    last_row_pwms = n_pwms % ncols if n_pwms % ncols != 0 else ncols
    shift_frac = (ncols - last_row_pwms) / (2 * ncols) - 0.014

    for i, (pwm, trials) in enumerate(all_trials_per_pwm.items()):
        row = i // ncols
        col = i % ncols

        ax = axes[i]
        ax.axis("on")

        # Shift last row axes if not filled
        if row == nrows - 1 and last_row_pwms < ncols:
            pos = ax.get_position()
            new_x0 = pos.x0 + shift_frac
            ax.set_position([new_x0, pos.y0, pos.width, pos.height])

        # Overlay trials for this PWM
        for trial in trials:
            x_data = trial["x_data"]
            y_data = trial["y_data"]
            xc, yc, R = trial["xc"], trial["yc"], trial["R"]

            ax.plot(x_data, y_data, "steelblue", lw=0.8, alpha=0.8)
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(xc + R * np.cos(theta),
                    yc + R * np.sin(theta),
                    "r--", lw = 0.6, alpha = 0.6)
            ax.set_xticks([0,100,200])
            ax.set_yticks([0,100,200])


        ax.set_title(rf"$\textbf{{{pwm}}}$")

        # Selective labeling
        if row == nrows - 1:
            ax.set_xlabel(r"$x$ (cm)")
        if col == 0:
            ax.set_ylabel(r"$y$ (cm)")


    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="steelblue", lw = 0.8, alpha = 1, label="Trajectory"),
            plt.Line2D([0], [0], color="red", lw=0.6, alpha = 0.6, ls="--", label="Fitted circle")
        ],
        loc="upper center", ncol=2, frameon=False
    )

    if save_path:
        plt.savefig(save_path + ".pdf", dpi=400, format="pdf")
    plt.close(fig)



def plot_msd_all(all_trials_per_pwm: dict,
                 save_path: str = None):
    
    """
    Generate a single figure overlaying MSD curves for all PWM values.

    - Each PWM is assigned a distinct color (from colormap).
    - Multiple trials per PWM are shown as faint lines.
    - The average MSD curve across trials (per PWM) is shown as a thicker line.
    - A colorbar indicates PWM values instead of a legend.

    Parameters
    ----------
    all_trials_per_pwm (dict):
        dictionary mapping PWM values to lists of trial_data dicts.
        Each trial_data should come from `compute_v_msd` and contain
        {"taus_sec_squared", "msd"}.

    save_path (str or None):
        if provided, saves the figure to this path (PDF).
        Otherwise, shows the figure interactively.
    """

    n_pwms = len(all_trials_per_pwm)
    cmap = cm.get_cmap("Blues", n_pwms)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xticks([0, 1, 2])

    # For colorbar
    pwm_values = sorted(all_trials_per_pwm.keys())
    norm = plt.Normalize(vmin=min(pwm_values), vmax=max(pwm_values))

    for _, (pwm, trials_data) in enumerate(sorted(all_trials_per_pwm.items())):
        color = cmap(norm(pwm))  # map PWM to color

        taus_common = None
        msd_list = []

        for trial_data in trials_data:
            taus = trial_data["taus_sec_squared"]
            msd = trial_data["msd"]

            if taus_common is None:
                taus_common = taus

            # Plot raw MSD (faint lines)
            ax.plot(taus, msd, color=color, alpha=0.5, lw=0.5)

            msd_list.append(msd)

        if len(msd_list) > 0:
            msd_mean = np.mean(msd_list, axis=0)
            ax.plot(taus_common, msd_mean, color=color, lw=1.7, alpha=1)

    ax.set_xlabel(r"$\tau^2$ (s$^2$)")
    ax.set_ylabel(r"MSD (cm$^2$)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("PWM")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path + ".pdf", dpi=400, format="pdf")
    else:
        plt.show()

    plt.close(fig)
