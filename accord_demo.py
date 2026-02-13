# pylint: disable=protected-access, too-many-locals, too-many-statements, broad-exception-caught, too-many-nested-blocks, too-many-branches
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

import asyncio
import math
import os
from typing import Optional
import numpy as np
from src.plotting import plot_constellation, \
    plot_nis_consistency_by_satellite, \
    plot_reputation, check_consensus_outcomes, \
        plot_nis_boxplot
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG
from src.filter import FilterConfig, \
    simulate_truth_and_meas, JointEKF, ObservationRecord
from src.logger import get_logger
from src.satellite_node import SatelliteNode

logger = get_logger()

# Maximum range for Inter-Satellite Links (ISL) in meters
ISL_RANGE_METERS = 4000e3  # 4000 km

def clear_log() -> None:
    """
    Clear the application log file at the start of the demo.
    """
    log_file_path = "app.log"
    if os.path.exists(log_file_path):
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.truncate(0)
        logger.info("Cleared app.log at the start of accord_demo.py")


def is_in_isl_range(sat1: SatelliteNode, sat2: SatelliteNode) -> bool:
    """
    Checks if two satellites are within ISL range of each other.

    Args:
    - sat1: The first SatelliteNode.
    - sat2: The second SatelliteNode.

    Returns:
    - True if the satellites are within range, False otherwise.
    """
    distance = math.sqrt(
        (sat1.x - sat2.x)**2 +
        (sat1.y - sat2.y)**2 +
        (sat1.z - sat2.z)**2
    )
    return distance <= ISL_RANGE_METERS

async def run_consensus_demo(config: FilterConfig,
                             save_ekf_results: bool = True,
                             load_ekf_results: bool = False,
                             ekf_results_path: str = \
                                "sim_data/ekf_simulation_results.npz") -> \
                                    tuple[Optional[DAG],
                                          Optional[dict],
                                          Optional[np.ndarray]]:
    """
    Run a demo of the consensus mechanism with multiple satellite nodes
    submitting transactions to the DAG.

    Args:
    - config: FilterConfig object with simulation parameters.
    - save_ekf_results: If True, saves the EKF simulation results to ekf_results_path.
    - load_ekf_results: If True, attempts to load EKF simulation results from ekf_results_path.
                        If successful, skips the EKF simulation phase.
    - ekf_results_path: Path to the .npz file for saving/loading EKF results.

    Returns:
    - A tuple containing:
        - The final DAG object after all transactions have been processed.
        - A dictionary containing the reputation history for each satellite.
        - The ground truth trajectory history.
    """
    clear_log()

    truth = None
    z_hist = None
    all_obs_records: Optional[list[ObservationRecord]] = None
    x_hist = None

    # Attempt to load EKF results if requested
    if load_ekf_results and os.path.exists(ekf_results_path):
        logger.info("Attempting to load EKF simulation results from %s", ekf_results_path)
        try:
            with np.load(ekf_results_path, allow_pickle=True) as data:
                # Reconstruct FilterConfig from saved attributes
                loaded_config = FilterConfig(
                    N=data['config_N'],
                    steps=data['config_steps'],
                    dt=data['config_dt'],
                    sig_r=data['config_sig_r'],
                    sig_rdot=data['config_sig_rdot'],
                    q_acc_target=data['config_q_acc_target'],
                    q_acc_obs=data['config_q_acc_obs'],
                    seed=data['config_seed']
                )

                # Check if loaded config matches current config
                if loaded_config == config:
                    truth = data['truth']
                    z_hist = data['z_hist']
                    # Ensure all_obs_records is converted back to a list of dataclasses
                    all_obs_records = data['all_obs_records']
                    x_hist = data['x_hist']
                    logger.info("Successfully loaded EKF simulation results.")
                else:
                    logger.warning("Loaded EKF config does not match current config. \
                        Rerunning EKF simulation.")
        except Exception as e:
            logger.error("Failed to load EKF simulation results: %s. Rerunning EKF simulation.", e)

    # If EKF results were not loaded or loading failed, run the EKF simulation
    if truth is None or z_hist is None or all_obs_records is None or x_hist is None:
        logger.info("Simulating satellite constellation to get truth")
        truth, z_hist = simulate_truth_and_meas(
            config.N, config.steps, config.dt, config.sig_r,
            config.sig_rdot, config.seed
        )

        logger.info("Initializing Joint EKF")
        ekf = JointEKF(config, truth[0])

        logger.info("Collecting observation records")
        all_obs_records = []
        x_hist = np.zeros((config.steps, config.N * 6))
        for k in range(config.steps):
            logger.info("Starting prediction")
            ekf.predict()
            logger.info("Starting update")
            obs_records_step = ekf.update(z_hist[k], k)
            logger.info("Adding new record")
            all_obs_records.extend(obs_records_step)
            x_hist[k] = ekf.ekf.x
            logger.info("Completed EKF step %d/%d", k + 1, config.steps)

        # Save EKF results if requested
        if save_ekf_results:
            logger.info("Saving EKF simulation results to %s", ekf_results_path)
            os.makedirs(os.path.dirname(ekf_results_path), exist_ok=True)
            np.savez_compressed(
                ekf_results_path,
                config_N=config.N,
                config_steps=config.steps,
                config_dt=config.dt,
                config_sig_r=config.sig_r,
                config_sig_rdot=config.sig_rdot,
                config_q_acc_target=config.q_acc_target,
                config_q_acc_obs=config.q_acc_obs,
                config_seed=config.seed,
                truth=truth,
                z_hist=z_hist,
                all_obs_records=np.array(all_obs_records, dtype=object), # Save as object array
                x_hist=x_hist
            )
            logger.info("EKF simulation results saved successfully.")

    # Ensure data is available for the consensus part
    if all_obs_records is None or x_hist is None or truth is None:
        logger.error("EKF simulation data is not available after loading or running. Exiting.")
        return None, None, None

    poise = ConsensusMechanism()
    queue: asyncio.Queue = asyncio.Queue()
    dag = DAG(queue=queue, consensus_mech=poise)

    asyncio.create_task(dag.listen())

    # Create one SatelliteNode per unique observer_id in the JSON
    unique_ids = sorted(list(range(config.N)))
    satellites: dict[int, SatelliteNode] = {
        sid: SatelliteNode(node_id=sid, queue=queue) for sid in unique_ids
    }
    rep_history: dict[str, list[float]] = {str(sid): [] for sid in unique_ids}

    # Initialise rep_history with the starting reputation for all satellites
    for sid in unique_ids:
        rep_history[str(sid)].append(satellites[sid].reputation)

    # Group observations by step
    obs_by_step: dict[int, list[ObservationRecord]] = {i: [] for i in range(config.steps)}
    for obs in all_obs_records:
        obs_by_step[obs.step].append(obs)

    # Define satellite IDs for special behavior
    perfect_sat_id = 1
    faulty_sat_id = 2
    intermittent_sat_id = 3

    for k in range(config.steps):
        # Update satellite positions at each step
        for sid, sat in satellites.items():
            state_vector = truth[k, sid*6:(sid+1)*6]
            sat.update_position(state_vector)

        transactions_submitted_this_step = {sid: False for sid in unique_ids}

        # Iterate through satellites to check for ISL opportunities
        for sid, sat in satellites.items():
            for other_sid, other_sat in satellites.items():
                if sid == other_sid:
                    continue

                if is_in_isl_range(sat, other_sat):
                    # Find the corresponding observation record
                    obs_to_submit = None
                    for obs in obs_by_step.get(k, []):
                        if obs.observer == sid and obs.target == other_sid:
                            obs_to_submit = obs
                            break

                    if obs_to_submit:
                        # --- Inject special satellite behavior ---
                        if sid == perfect_sat_id:
                            obs_to_submit.nis = 0.01
                        elif sid == faulty_sat_id and config.N >= 7:
                            obs_to_submit.nis = 50.0
                        elif sid == intermittent_sat_id and config.N >= 10:
                            if 200 <= k < 400:
                                if obs_to_submit.nis > 2.0:
                                    obs_to_submit.nis = obs_to_submit.nis * 10
                                else:
                                    obs_to_submit.nis = obs_to_submit.nis / 10

                        sat.load_sensor_data(obs_to_submit)
                        logger.info("Satellite %s: submitting transaction \
                                    for witness of %s.", sid, other_sid)
                        await sat.submit_transaction(recipient_address=other_sid)
                        transactions_submitted_this_step[sid] = True

            # If no transaction submitted, reputation decays towards neutral
            if not transactions_submitted_this_step[sid]:
                sat.reputation = sat.rep_manager.decay(sat.reputation)

        # Record reputation for all satellites at the end of the step
        for sid in unique_ids:
            sat = satellites[sid]
            rep_history[str(sid)].append(sat.reputation)

    return dag, rep_history, truth

# Run demo
if __name__ == "__main__":
    default_config = FilterConfig(
        N=50,
        steps=1000,
        dt=60.0,
        sig_r=10.0,
        sig_rdot=0.2,
        q_acc_target=1e-5,
        q_acc_obs=1e-5,   # kept for signature compatibility
        seed=42,
    )

    DATA_DIR = "sim_data"
    DATA_FILENAME = "ekf_simulation_results.npz"
    RESULTS_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

    final_dag, rep_hist, truth_history= asyncio.run(
        run_consensus_demo(default_config, load_ekf_results=True, ekf_results_path=RESULTS_PATH)
    )

    # Use the results from the loaded run for plotting
    if final_dag:
        plot_nis_consistency_by_satellite(final_dag)
        plot_nis_boxplot(final_dag)
        check_consensus_outcomes(final_dag)
    if rep_hist:
        plot_reputation(rep_hist)
    if truth_history is not None:
        plot_constellation(truth_history, default_config.N)
