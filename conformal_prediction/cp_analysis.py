# pylint: disable= too-many-locals, too-many-statements
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Constants for Analysis ---
INITIAL_REP = 0.5
REWARD = 0.01
PUNISHMENT = 0.05
MAX_REP = 1.0
MIN_REP = 0.0

def cp_analysis():
    """
    Analyzes NIS history using Conformal Prediction to generate pseudo-reputation profiles.
    """
    # 1. Read the nis_history.csv file
    print("Reading nis_history.csv...")
    try:
        df = pd.read_csv('conformal_prediction/nis_history.csv')
    except FileNotFoundError:
        print("Error: nis_history.csv not found. Please run the extraction script first.")
        return

    # ==============================================================================
    # TRANSLATION STEP 1 & 2: The "Model and Calibration Set
    # Our "model" is the EKF orbital simulation. We don't train it here; we just
    # extract its outputs. Our "calibration set" isn't a random slice, but specifically
    # the stable, post-convergence data (Timesteps 200+) from known Honest satellites.
    # ==============================================================================
    calibration_data = df[(df['satellite_id'] % 10 > 3) | (df['satellite_id'] % 10 == 0)]
    calibration_data = calibration_data[(calibration_data['timestep'] >= 200)]

    if calibration_data.empty:
        print("Error: No calibration data found for honest satellites in the first 200 timesteps.")
        return

    # ==============================================================================
    # TRANSLATION STEP 3: Nonconformity Scores
    # We do not need to calculate the difference because the Normalized Innovation
    # Squared (NIS) is our nonconformity score. It is the mathematical difference
    # between the EKF prediction and the actual measurement.
    # ==============================================================================
    scores = calibration_data['nis_value'].values
    n = len(scores)
    alpha = 0.05  # 95% significance level

    # ==============================================================================
    # TRANSLATION STEP 4: The Prediction Interval (Bounds)
    # Instead of a simple one-sided bound, we use a robust TWO-SIDED bound to catch
    # both noisy hardware (upper bound) and malicious spoofers (lower bound).
    # ==============================================================================
    lower_bound = np.percentile(scores, (alpha / 2) * 100)
    upper_bound = np.percentile(scores, (1 - alpha / 2) * 100)

    print(f"Calibration finished. n={n}, alpha={alpha}")
    print(f"CP Bounds for NIS: [{lower_bound:.4f}, {upper_bound:.4f}]")

    # 4. Loop through the dataset and calculate pseudo-reputation scores
    satellites = df['satellite_id'].unique()

    # Define behavior profiles
    groups = {
        'Spoofers': [sid for sid in satellites if sid % 10 == 1],
        'Faulty': [sid for sid in satellites if sid % 10 == 2],
        'Intermittent': [sid for sid in satellites if sid % 10 == 3],
        'Honest': [sid for sid in satellites if sid % 10 not in [1, 2, 3]]
    }

    reputations = {sid: INITIAL_REP for sid in satellites}
    history = {group: [] for group in groups}

    # Start analysis
    analysis_steps = list(sorted(df['timestep'].unique()))

    print(f"Analyzing {len(analysis_steps)} timesteps...")

    for t in analysis_steps:
        step_data = df[df['timestep'] == t]

        for _, row in step_data.iterrows():
            sid = row['satellite_id']
            nis = row['nis_value']

            # ==========================================================================
            # TRANSLATION STEP 5: Live Prediction / Assessment
            # Actual NIS can be assessed relative to the
            # CP covariance to determine the 'current' and 'future' reputation score.
            # ==========================================================================
            if lower_bound <= nis <= upper_bound:
                # Reward (Conforming)
                reputations[sid] = min(MAX_REP, reputations[sid] + REWARD)
            else:
                # Punish (Non-conforming)
                reputations[sid] = max(MIN_REP, reputations[sid] - PUNISHMENT)

        # Calculate averages for each group at this step
        for group_name, group_sids in groups.items():
            if not group_sids:
                history[group_name].append(INITIAL_REP)
                continue

            group_reps = [reputations[sid] for sid in group_sids]
            history[group_name].append(np.mean(group_reps))

    plot_df = pd.DataFrame(history)
    plot_df['Timestep'] = analysis_steps
    plot_df = plot_df[['Timestep', 'Spoofers', 'Faulty', 'Intermittent', 'Honest']]
    plot_df.to_csv('conformal_prediction/plot_data.csv', index=False)

    # 5. Plot the average reputation scores
    plt.figure(figsize=(12, 7))

    plot_styles = {
        'Spoofers': {'color': 'red', 'linestyle': '-', 'linewidth': 4},
        'Faulty': {'color': 'orange', 'linestyle': '--', 'linewidth': 2},
        'Intermittent': {'color': 'magenta', 'linestyle': '-', 'linewidth': 2},
        'Honest': {'color': 'green', 'linestyle': '-', 'linewidth': 2}
    }

    for group_name, reps in history.items():
        style = plot_styles[group_name]
        plt.plot(
            analysis_steps,
            reps,
            label=group_name,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth']
        )

    plt.axhline(y=INITIAL_REP, color='gray', linestyle='-.', label='Neutral Reputation')
    plt.axvline(x=200, color='black', linestyle='--', label='Filter convergence point')
    plt.xlabel('Timestep')
    plt.ylabel('Average Pseudo-Reputation')
    plt.title('CP-based Pseudo-Reputation Analysis by Behavior Profile')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    output_plot = 'conformal_prediction/cp_reputation_profiles.png'
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    cp_analysis()
