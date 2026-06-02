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
import shutil
from typing import Optional
import numpy as np
from src.plotting import  \
    plot_aggregated_reputation, check_consensus_outcomes, \
        plot_nis_boxplot, plot_ground_tracks, \
            calculate_convergence_index, \
                calculate_nis_convergence_index, \
                    calculate_median_percentiles, \
                        plot_constellation
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG, MockDAG
from src.filter import FilterConfig, \
    simulate_truth_and_meas, JointEKF, ObservationRecord, \
    apply_network_faults
from src.logger import get_logger
from src.satellite_node import SatelliteNode
#------------------
# Constants
CLUSTER_SIZE = 10

DATA_DIR = "sim_data"
DATA_FILENAME = "ekf_simulation_results.npz"
EKF_RESULTS_PATH = os.path.join(DATA_DIR, DATA_FILENAME)

SIM_RESULTS_PATH = os.path.join(DATA_DIR, "sim_results.npz")

DEFAULT_CONFIG = FilterConfig(
        N=400,
        steps=1000,
        dt=60.0,
        sig_r=10.0,
        sig_rdot=0.2,
        q_acc_target=1e-5,
        seed=42,
        ISL_range_m=1000e3
    )
#------------------
# Helper functions


def clear_log(log_file_path: str = "app.log") -> None:
    """
    Clear the application log file at the start of the demo.
    """
    if os.path.exists(log_file_path):
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.truncate(0)
        get_logger().info("Cleared %s at the start of accord_demo.py", log_file_path)


def is_in_isl_range(isl_range: float, sat1: SatelliteNode, sat2: SatelliteNode) -> bool:
    """
    Checks if two satellites are within ISL range of each other.

    Args:
    - isl_range: The range for inter-satellite communication in metres.
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
    return distance <= isl_range

#------------------
# Main demo function

async def run_consensus_demo(config: FilterConfig,
                             save_ekf_results: bool = True,
                             load_ekf_results: bool = False,
                             ekf_results_path: str = \
                                "sim_data/ekf_simulation_results.npz",
                             clear_logs: bool = True,
                             log_file: str = "app.log",
                             save_sim_results: bool = True,
                             run_consensus: bool = True,
                             sim_results_path: str = SIM_RESULTS_PATH) -> \
        tuple[Optional[DAG], Optional[dict], Optional[np.ndarray], Optional[set[int]]]:
    """
    Run a demo of the consensus mechanism with multiple satellite nodes
    submitting transactions to the DAG.

    Args:
    - config: FilterConfig object with simulation parameters.
    - save_ekf_results: If True, saves the EKF simulation results to ekf_results_path.
    - load_ekf_results: If True, attempts to load EKF simulation results from ekf_results_path.
                        If successful, skips the EKF simulation phase.
    - ekf_results_path: Path to the .npz file for saving/loading EKF results.
    - clear_logs: If True, clears the app.log file at the start.
    - log_file: The file to write logs to.
    - save_sim_results: If True, saves the final consensus simulation results to
                        sim_results_path.
    - run_consensus: If False, returns early after EKF simulation phase.
    - sim_results_path: Path to save consensus simulation results.

    Returns:
    - A tuple containing:
        - The final DAG object after all transactions have been processed.
        - A dictionary containing the reputation history for each satellite.
        - The ground truth trajectory history.
        - A set of faulty satellite IDs.
    """
    logger = get_logger(log_file=log_file)
    if clear_logs:
        clear_log(log_file)

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
                    seed=data['config_seed'],
                    ISL_range_m=data['config_ISL_range_m']
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
        except (OSError, ValueError) as e:
            logger.error(
                "Corrupt or unreadable EKF results file. Rerunning EKF simulation. Error: %s", e
            )
        except KeyError as e:
            logger.error(
                "EKF results file is missing expected key %s. Rerunning EKF simulation.", e
            )

    # If EKF results were not loaded or loading failed, run the EKF simulation
    if truth is None or z_hist is None or all_obs_records is None or x_hist is None:
        logger.info("Simulating satellite constellation to get truth")
        truth, z_hist = simulate_truth_and_meas(
            config.N, config.steps, config.dt, config.sig_r,
            config.sig_rdot, config.seed
        )

        # --- Start of Clustered EKF Implementation ---
        logger.info("Initialising Clustered EKF with cluster size %s", CLUSTER_SIZE)

        # 1. Create clusters of satellite IDs
        all_sat_ids = list(range(config.N))
        clusters = [
            all_sat_ids[i:i + CLUSTER_SIZE]
            for i in range(0, config.N, CLUSTER_SIZE)
        ]

        # 2. Create a list of EKF instances, one for each cluster
        cluster_ekfs = []
        for i, cluster_sat_ids in enumerate(clusters):
            cluster_n = len(cluster_sat_ids)
            cluster_config = FilterConfig(
                N=cluster_n,
                steps=config.steps,
                dt=config.dt,
                sig_r=config.sig_r,
                sig_rdot=config.sig_rdot,
                q_acc_target=config.q_acc_target,
                seed=config.seed + i, # Use different seed for each cluster
                ISL_range_m=config.ISL_range_m
            )

            # Extract initial truth state for this cluster
            # Extract initial truth state for this cluster
            initial_state_slices = [truth[0, sat_id*6:(sat_id+1)*6] for sat_id in cluster_sat_ids]
            cluster_truth_0 = np.concatenate(initial_state_slices)

            cluster_ekfs.append(JointEKF(cluster_config, cluster_truth_0))
            logger.info("Initialised EKF for cluster %d with %d satellites: %s",
                        i, cluster_n, cluster_sat_ids)

        # 3. Pre-calculate the mapping from (observer, target) to z_hist index
        z_map = {}
        z_idx = 0
        for i in range(config.N):
            for j in range(config.N):
                if i != j:
                    z_map[(i, j)] = slice(z_idx, z_idx + 2)
                    z_idx += 2

        # 4. Main simulation loop
        logger.info("Collecting observation records using Clustered EKF")
        all_obs_records = []
        x_hist = np.zeros((config.steps, config.N * 6))

        for k in range(config.steps):
            for cluster_sat_ids, ekf in zip(clusters, cluster_ekfs):
                # Predict step for the cluster
                ekf.predict()

                # Build the measurement vector `z_k_cluster` for this cluster
                z_k_cluster_list = []
                for obs_id_global in cluster_sat_ids:
                    for tgt_id_global in cluster_sat_ids:
                        if obs_id_global == tgt_id_global:
                            continue

                        z_slice = z_map.get((obs_id_global, tgt_id_global))
                        if z_slice:
                            z_k_cluster_list.append(z_hist[k, z_slice])

                if not z_k_cluster_list:
                    continue

                z_k_cluster = np.concatenate(z_k_cluster_list)

                # Update step for the cluster
                obs_records_step = ekf.update(z_k_cluster, k)

                # Remap local satellite IDs in records to global IDs and store
                for record in obs_records_step:
                    record.observer = cluster_sat_ids[record.observer]
                    record.target = cluster_sat_ids[record.target]
                all_obs_records.extend(obs_records_step)

                # Update the global state history `x_hist`
                for sat_idx_local, sat_idx_global in enumerate(cluster_sat_ids):
                    global_slice = slice(sat_idx_global * 6, (sat_idx_global + 1) * 6)
                    local_slice = slice(sat_idx_local * 6, (sat_idx_local + 1) * 6)
                    x_hist[k, global_slice] = ekf.ekf.x[local_slice]

            logger.info("Completed EKF step %d/%d", k + 1, config.steps)
        # --- End of Clustered EKF Implementation ---

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
                config_seed=config.seed,
                config_ISL_range_m=config.ISL_range_m,
                truth=truth,
                z_hist=z_hist,
                all_obs_records=np.array(all_obs_records, dtype=object), # Save as object array
                x_hist=x_hist
            )
            logger.info("EKF simulation results saved successfully.")

    # Early return if only EKF was requested
    if not run_consensus:
        logger.info("run_consensus is False. Returning early after EKF phase.")
        return None, None, truth, None

    # Ensure data is available for the consensus part
    if all_obs_records is None or x_hist is None or truth is None:
        logger.error("EKF simulation data is not available after loading or running. Exiting.")
        return None, None, None, None

    faulty_ids: set[int] = set()
    poise = ConsensusMechanism()
    queue: asyncio.Queue = asyncio.Queue()
    dag = DAG(queue=queue, consensus_mech=poise)

    listen_task = asyncio.create_task(dag.listen())

    try:
        # Create one SatelliteNode for each of the N satellites in the simulation.
        unique_ids = sorted(list(range(config.N)))
        satellites: dict[int, SatelliteNode] = {
            sid: SatelliteNode(node_id=sid, queue=queue) for sid in unique_ids
        }
        rep_history: dict[str, list[float]] = {str(sid): [] for sid in unique_ids}

        # Initialise rep_history with the starting reputation for all satellites.
        for sid in unique_ids:
            rep_history[str(sid)].append(satellites[sid].reputation)

        # Group observations by step
        obs_by_step: dict[int, list[ObservationRecord]] = {i: [] for i in range(config.steps)}
        for obs in all_obs_records:
            obs_by_step[obs.step].append(obs)

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

                    if is_in_isl_range(config.ISL_range_m, sat, other_sat):
                        # Find the corresponding observation record
                        obs_to_submit = None
                        for obs in obs_by_step.get(k, []):
                            if obs.observer == sid and obs.target == other_sid:
                                obs_to_submit = obs
                                break

                        if obs_to_submit:
                            # Inject dishonest behaviour profiles
                            apply_network_faults(obs_to_submit, sid, config.N, k, faulty_ids)

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

        # Save Consensus Simulation results
        if save_sim_results:
            logger.info("Saving Simulation results to %s", sim_results_path)
            os.makedirs(os.path.dirname(sim_results_path), exist_ok=True)
            np.savez_compressed(
                sim_results_path,
                dag_ledger=dag.ledger, # type: ignore[arg-type]
                rep_history=rep_history, # type: ignore[arg-type]
                truth=truth,
                faulty_ids=np.array(list(faulty_ids))
            )
            logger.info("Simulation results saved successfully.")

    finally:
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass

    return dag, rep_history, truth, faulty_ids

# Run demo
if __name__ == "__main__":
    accord_logger = get_logger()

    FINAL_DAG: DAG | MockDAG | None = None
    REP_HIST: Optional[dict] = None
    TRUTH: Optional[np.ndarray] = None
    FAULTY_IDS: Optional[set[int]] = None

    # Attempt to load simulation results if they exist
    if os.path.exists(SIM_RESULTS_PATH):
        accord_logger.info("Attempting to load simulation results from %s", SIM_RESULTS_PATH)
        try:
            with np.load(SIM_RESULTS_PATH, allow_pickle=True) as simulated_data:
                # pylint: disable=no-member
                # Check if the number of satellites in the saved data matches the current config
                saved_N = int(simulated_data['truth'].shape[1] / 6)
                if saved_N == DEFAULT_CONFIG.N:
                    dag_ledger = simulated_data['dag_ledger'].item()
                    FINAL_DAG = MockDAG(dag_ledger)
                    REP_HIST = simulated_data['rep_history'].item()
                    TRUTH = simulated_data['truth']
                    FAULTY_IDS = set(simulated_data['faulty_ids'])
                    accord_logger.info("Successfully loaded Simulation results.")
                else:
                    accord_logger.warning(
                        "Loaded config (N=%d) does not match current config (N=%d). "
                        "Rerunning simulation.",
                        saved_N, DEFAULT_CONFIG.N
                    )
        except (OSError, ValueError) as e:
            accord_logger.error(
                "Corrupt or unreadable simulation results file. Rerunning simulation. Error: %s", e
            )
        except KeyError as e:
            accord_logger.error(
                "Simulation results file is missing expected data array %s. \
                    Rerunning simulation.", e
            )

    # If simulation results were not loaded or loading failed, run the consensus simulation
    if TRUTH is None or REP_HIST is None or FINAL_DAG is None or FAULTY_IDS is None:
        FINAL_DAG, REP_HIST, TRUTH, FAULTY_IDS = asyncio.run(
            run_consensus_demo(DEFAULT_CONFIG, load_ekf_results=True,
            ekf_results_path=EKF_RESULTS_PATH))

        # Copy the log file to the sim_data directory
        if os.path.exists("app.log"):
            shutil.copy("app.log", os.path.join(DATA_DIR, "app.log"))
            accord_logger.info("Copied app.log to %s.", DATA_DIR)

    # Use the results for plotting
    if FINAL_DAG is not None and FAULTY_IDS is not None:
        plot_nis_boxplot(FINAL_DAG, faulty_ids=FAULTY_IDS)
        NIS_CONVERGENCE_INDEX = calculate_nis_convergence_index(FINAL_DAG,\
            faulty_ids=FAULTY_IDS)
        plot_nis_boxplot(FINAL_DAG, faulty_ids=FAULTY_IDS, \
            convergence_index=NIS_CONVERGENCE_INDEX)
        calculate_median_percentiles()
        check_consensus_outcomes(FINAL_DAG)
    if REP_HIST and FAULTY_IDS is not None:
        CONVERGENCE_IDX = calculate_convergence_index(REP_HIST, faulty_ids=FAULTY_IDS)
        plot_aggregated_reputation(REP_HIST, faulty_ids=FAULTY_IDS,
                                   start_at_full_constellation=False,
                                   convergence_index=CONVERGENCE_IDX)
    if TRUTH is not None and FAULTY_IDS is not None:
        plot_ground_tracks(TRUTH, DEFAULT_CONFIG.N)
        plot_constellation(TRUTH, DEFAULT_CONFIG.N)
