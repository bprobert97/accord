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


def visualise_orbits(data_path='sim_data/sim_results.npz',
                     output_gif='images/orbits.gif', frames=200, interval=50) -> None:
    """
    Creates an animated GIF of satellite orbits around Earth.

    Args:
        data_path: Path to the .npz file containing 'truth' data.
        output_gif: Path to save the resulting GIF.
        frames: Number of frames to animate (subsamples the data if less than total steps).
        interval: Delay between frames in milliseconds. Default 50ms for smoother motion.

    Returns:
        None. Saves the animation as a GIF at the specified location.
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    # Load data
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    truth = data['truth'] # Shape: (steps, 6*N)

    steps, state_size = truth.shape
    n_sats = state_size // 6
    print(f"Loaded {n_sats} satellites over {steps} time steps.")

    # Subsample frame indices
    frames = min(frames, steps)

    indices = np.linspace(0, steps - 1, frames, dtype=int)

    # Setup 3D Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Earth model
    r_e = 6378e3  # Earth radius in meters
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_earth = r_e * np.outer(np.cos(u), np.sin(v))
    y_earth = r_e * np.outer(np.sin(u), np.sin(v))
    z_earth = r_e * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_earth, y_earth, z_earth, color='blue',  # type: ignore[attr-defined]
                    alpha=0.1, rstride=2, cstride=2)

    # Initialise satellite markers
    dots = []
    trails = []
    trail_len = 30 # Number of animation steps to show as a tail

    # Randomly select up to 50 satellites to visualise if N is too large for clear animation
    viz_n = min(n_sats, 50)
    viz_indices = np.random.choice(range(n_sats), viz_n, replace=False)

    # pylint: disable=no-member
    colors = plt.cm.viridis(np.linspace(0, 1, viz_n))  # type: ignore[attr-defined]

    for i in range(viz_n):
        dot, = ax.plot([], [], [], 'o', color=colors[i], markersize=4)
        dots.append(dot)
        trail, = ax.plot([], [], [], '-', color=colors[i], alpha=0.4, linewidth=1.5)
        trails.append(trail)

    # Add text overlays
    _ = ax.text2D(0.05, 0.95, f"Satellites shown: {viz_n} / {n_sats}",  # type: ignore[attr-defined]
                           transform=ax.transAxes, fontsize=12, fontweight='bold')
    step_text = ax.text2D(0.05, 0.90, "", transform=ax.transAxes, # type: ignore[attr-defined]
                          fontsize=12)

    # Set plot limits
    # Calculate limits from first step positions to ensure Earth and orbits are visible
    max_dist = np.max(np.linalg.norm(truth[0, :3], axis=0))
    limit = max_dist * 1.2 if max_dist > 0 else r_e * 2

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)  # type: ignore[attr-defined]
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')  # type: ignore[attr-defined]
    ax.set_title('Satellite Constellation Animation', fontsize=16)

    def init():
        for dot, trail in zip(dots, trails):
            dot.set_data([], [])
            dot.set_3d_properties([])
            trail.set_data([], [])
            trail.set_3d_properties([])
        step_text.set_text("")
        return dots + trails + [step_text]

    def update(frame):
        # Update each satellite
        actual_step = indices[frame]

        # Calculate start index for trail in terms of ACTUAL steps to ensure smoothness
        start_frame_idx = max(0, frame - trail_len)
        start_step = indices[start_frame_idx]

        for i, sat_idx in enumerate(viz_indices):
            # Current position
            pos = truth[actual_step, sat_idx*6 : sat_idx*6+3]
            dots[i].set_data([pos[0]], [pos[1]])
            dots[i].set_3d_properties([pos[2]])

            # Trail using full resolution data between start_step and actual_step
            trail_pos = truth[start_step : actual_step + 1, sat_idx*6 : sat_idx*6+3]
            if len(trail_pos) > 0:
                trails[i].set_data(trail_pos[:, 0], trail_pos[:, 1])
                trails[i].set_3d_properties(trail_pos[:, 2])

        # Update step counter
        step_text.set_text(f"Timestep: {actual_step}")

        return dots + trails + [step_text]

    print("Creating animation...")
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=interval)

    print(f"Saving to {output_gif}...")
    writer = PillowWriter(fps=1000 // interval)
    ani.save(output_gif, writer=writer)
    print("Done!")

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
