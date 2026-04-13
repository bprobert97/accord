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
import csv
import os
import numpy as np

# Define file paths
DATA_DIR = "sim_data"
CSV_DIR = "conformal_prediction"
EKF_RESULTS_PATH = os.path.join(DATA_DIR, "ekf_simulation_results.npz")
SIM_RESULTS_PATH = os.path.join(DATA_DIR, "sim_results.npz")
OUTPUT_CSV = os.path.join(CSV_DIR, "nis_history.csv")

def extract_nis_to_csv():
    """
    Extracts NIS values and metadata from simulation results and exports them to a CSV file.
    """
    if not os.path.exists(EKF_RESULTS_PATH):
        print(f"Error: {EKF_RESULTS_PATH} not found.")
        return

    if not os.path.exists(SIM_RESULTS_PATH):
        print(f"Error: {SIM_RESULTS_PATH} not found.")
        return

    print(f"Loading data from {EKF_RESULTS_PATH}...")
    with np.load(EKF_RESULTS_PATH, allow_pickle=True) as data:
        all_obs_records = data['all_obs_records']

    print(f"Loading faulty IDs from {SIM_RESULTS_PATH}...")
    with np.load(SIM_RESULTS_PATH, allow_pickle=True) as data:
        faulty_ids = set(data['faulty_ids'])

    print(f"Extracting {len(all_obs_records)} records to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['satellite_id', 'is_faulty', 'timestep', 'nis_value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # pylint: disable=not-an-iterable
        for record in all_obs_records:
            # record is an ObservationRecord object (from src.filter)
            sid = record.observer
            step = record.step

            # Start with the clean NIS value from the EKF
            nis = record.nis

            # --- INJECT MALICIOUS BEHAVIOR ---
            # Match the logic exactly from accord_demo.py
            if sid % 10 == 1:
                nis = 0.01
            elif sid % 10 == 2:
                nis = 50.0
            elif sid % 10 == 3:
                if 200 <= step < 400:
                    if nis > 2.0:
                        nis = nis * 10.0
                    else:
                        nis = nis / 10.0
            # ---------------------------------

            is_faulty = 1 if sid in faulty_ids else 0

            writer.writerow({
                'satellite_id': sid,
                'is_faulty': is_faulty,
                'timestep': step,
                'nis_value': nis
            })

    print(f"Successfully exported data to {OUTPUT_CSV}.")

if __name__ == "__main__":
    extract_nis_to_csv()
