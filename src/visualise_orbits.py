"""
The Autonomous Cooperative Consensus Orbit Determination (ACCORD) framework.
Author: Beth Probert
Email: beth.probert@strath.ac.uk

Copyright (C) 2025 Applied Space Technology Laboratory

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def visualise_orbits(data_path: str = 'sim_data/sim_results.npz',
                     output_gif_path: str = 'images/orbits.gif',
                     frames: int = 200, interval: int = 50) -> None:
    """
    Visualises the orbits of satellites from a .npz simulation data file, and saves as a GIF.

    Args:
    - data_path: Path to the .npz file containing the simulation results.
                 Must contain a 'truth' array of shape (timesteps, num_sats*6)
                 with satellite state vectors.
    - output_gif_path: Path to save the output GIF animation.
    - frames: Number of frames to include in the animation (evenly spaced through the simulation).
    - interval: Time in milliseconds between frames in the animation.

    Returns:
    - None. Saves the animation as a GIF at the specified path.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    print(f"Loading data from {data_path}...")
    # Inline the load statement to save local variables
    truth = np.load(data_path, allow_pickle=True)['truth']

    _generate_orbit_animation(truth, output_gif_path, frames, interval)


def _generate_orbit_animation(truth: np.ndarray,
                              output_gif_path: str,
                              frames: int,
                              interval: int) -> None:
    """
    Helper function to configure and run the matplotlib 3D animation.

    Args:
    - truth: Array of data of the true simulated orbits of the satellites.
    - output_gif_path: Path to save the output GIF animation.
    - frames: Number of frames to include in the animation (evenly spaced through the simulation).
    - interval: Time in milliseconds between frames in the animation.

    Returns:
    - None. Generates the animation.
    """
    steps = len(truth)
    # Inline min(frames, steps) directly into linspace
    indices = np.linspace(0, steps - 1, min(frames, steps), dtype=int)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    _draw_earth(ax)

    # Bundle markers into a single tuple to drastically reduce outer scope variables
    anim_elements = _setup_animation_markers(ax, truth.shape[1] // 6)

    max_dist = np.max(np.linalg.norm(truth[0, :3], axis=0))
    _set_axes_limits(ax, max_dist * 1.2 if max_dist > 0 else 6378e3 * 2)

    def init() -> list:
        """
        Initialises the animation by clearing all satellite markers and trails,
        and resetting the step text.
        """
        # Unpack inside the closure so Pylint counts them as local to 'init' only
        dots, trails, step_text, _ = anim_elements

        for dot, trail in zip(dots, trails):
            dot.set_data([], [])
            dot.set_3d_properties([])
            trail.set_data([], [])
            trail.set_3d_properties([])

        step_text.set_text("")
        return dots + trails + [step_text]

    def update(frame: int) -> list:
        """
        Updates the positions of satellite markers and trails for the given frame index.
        """
        # Unpack inside the closure so Pylint counts them as local to 'update' only
        dots, trails, step_text, viz_indices = anim_elements

        actual_step = indices[frame]
        start_step = indices[max(0, frame - 30)]

        for i, sat_idx in enumerate(viz_indices):
            pos = truth[actual_step, sat_idx*6 : sat_idx*6+3]
            dots[i].set_data([pos[0]], [pos[1]])
            dots[i].set_3d_properties([pos[2]])

            trail_pos = truth[start_step : actual_step + 1, sat_idx*6 : sat_idx*6+3]
            if len(trail_pos) > 0:
                trails[i].set_data(trail_pos[:, 0], trail_pos[:, 1])
                trails[i].set_3d_properties(trail_pos[:, 2])

        step_text.set_text(f"Timestep: {actual_step}")
        return dots + trails + [step_text]

    print("Creating animation...")
    ani = FuncAnimation(fig, update, frames=len(indices),
                        init_func=init, blit=True, interval=interval)
    print(f"Saving to {output_gif_path}...")
    ani.save(output_gif_path, writer=PillowWriter(fps=1000 // interval))
    print("Done!")

def _draw_earth(ax: plt.Axes) -> None:
    """
    Draws a simple representation of Earth as a blue sphere on the given 3D axes.

    Args:
    - ax: A Matplotlib 3D axes object to draw on.

    Returns:
    - None. Adds the Earth representation to the provided axes.
    """
    r_e = 6378e3
    u, v = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
    x_earth = r_e * np.outer(np.cos(u), np.sin(v))
    y_earth = r_e * np.outer(np.sin(u), np.sin(v))
    z_earth = r_e * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth,# type: ignore[attr-defined]
                    color='blue', alpha=0.1, rstride=2, cstride=2)

def _setup_animation_markers(ax: plt.Axes, n_sats: int) -> tuple[list, list, plt.Text, np.ndarray]:
    """
    Sets up the Matplotlib artists for satellite markers and trails, and a text
    element for the timestep.

    Args:
    - ax: A Matplotlib 3D axes object to add the artists to.
    - n_sats: The total number of satellites in the simulation, used to determine
                how many to visualize (capped at 50 for clarity).

    Returns:
    - A tuple containing:
        - dots: A list of Matplotlib 3D scatter plot objects for the satellite markers.
        - trails: A list of Matplotlib 3D line plot objects for the satellite trails.
        - step_text: A Matplotlib text object for displaying the timestep.
        - viz_indices: A NumPy array of indices for the visualized satellites.
    """
    viz_n = min(n_sats, 50)
    viz_indices = np.random.choice(range(n_sats), viz_n, replace=False)
    colors = plt.cm.viridis(np.linspace(0, 1, viz_n))  # type: ignore[attr-defined] # pylint: disable=no-member

    dots, trails = [], []
    for i in range(viz_n):
        dot, = ax.plot([], [], [], 'o', color=colors[i], markersize=4)
        trail, = ax.plot([], [], [], '-', color=colors[i], alpha=0.4, linewidth=1.5)
        dots.append(dot)
        trails.append(trail)

    ax.text2D(0.05, 0.95, f"Satellites shown: {viz_n} / {n_sats}", # type: ignore[attr-defined]
              transform=ax.transAxes, fontsize=12, fontweight='bold')
    step_text = ax.text2D(0.05, 0.90, "",  # type: ignore[attr-defined]
                          transform=ax.transAxes, fontsize=12)
    return dots, trails, step_text, viz_indices

def _set_axes_limits(ax: plt.Axes, limit: float) -> None:
    """
    Sets the limits and labels for the 3D axes.

    Args:
    - ax: A Matplotlib 3D axes object to set the limits on.
    - limit: The maximum absolute value for the x, y, and z axes. The limits will
             be set to [-limit, limit] for each axis.
    Returns:
    - None. Modifies the provided axes in-place.
    """
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit) # type: ignore[attr-defined]
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)') # type: ignore[attr-defined]
    ax.set_title('Satellite Constellation Animation', fontsize=16)

if __name__ == "__main__":
    # Check for existing data files
    potential_files = [
        'sim_data/sim_results.npz',
        'sim_data/ekf_simulation_results.npz'
    ]

    CHOSEN_FILE = None
    for f in potential_files:
        if os.path.exists(f):
            CHOSEN_FILE = f
            break

    if CHOSEN_FILE:
        visualise_orbits(data_path=CHOSEN_FILE)
    else:
        print("Could not find any .npz simulation data in sim_data/.")
