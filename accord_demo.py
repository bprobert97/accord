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
from typing import Optional, Any
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

    # Attempt to load or generate EKF data
    truth, _, all_obs_records, x_hist = _resolve_ekf_phase(
        config, load_ekf_results, ekf_results_path, save_ekf_results, logger
    )

    # Early return if only EKF was requested
    if not run_consensus:
        logger.info("run_consensus is False. Returning early after EKF phase.")
        return None, None, truth, None

    # Ensure data is available for the consensus part
    if all_obs_records is None or x_hist is None or truth is None:
        logger.error("EKF simulation data is not available after loading or running. Exiting.")
        return None, None, None, None

    return await _run_consensus_phase(
        config, truth, all_obs_records, save_sim_results, sim_results_path, logger
    )


def _resolve_ekf_phase(config: FilterConfig, load_ekf_results: bool, ekf_results_path: str,
                       save_ekf_results: bool, logger: Any) -> \
        tuple[Optional[np.ndarray], Optional[np.ndarray],
              Optional[list[ObservationRecord]], Optional[np.ndarray]]:

    """
    Resolves the EKF phase by either loading results from a file or running the simulation.

    Args:
    - config: FilterConfig object with simulation parameters.
    - load_ekf_results: If True, attempts to load EKF results from ekf_results_path.
    - ekf_results_path: Path to the .npz file for loading/saving EKF results.
    - save_ekf_results: If True, saves EKF results to ekf_results_path after running simulation.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - truth: The ground truth trajectory history (or None if loading failed).
        - z_hist: The measurement history (or None if loading failed).
        - all_obs_records: The list of observation records (or None if loading failed).
        - x_hist: The EKF state estimate history (or None if loading failed).
    """

    truth, z_hist, all_obs_records, x_hist = None, None, None, None

    # Attempt to load EKF results if requested
    if load_ekf_results and os.path.exists(ekf_results_path):
        truth, z_hist, all_obs_records, x_hist = _load_ekf_results(ekf_results_path, config, logger)

    # If EKF results were not loaded or loading failed, run the EKF simulation
    if truth is None or z_hist is None or all_obs_records is None or x_hist is None:
        truth, z_hist, all_obs_records, x_hist = _simulate_ekf_filter(config, logger)

        # Save EKF results if requested
        if save_ekf_results:
            _save_ekf_results(ekf_results_path, config, truth,
                              z_hist, all_obs_records, x_hist, logger)

    return truth, z_hist, all_obs_records, x_hist


def _load_ekf_results(ekf_results_path: str, config: FilterConfig, logger: Any) -> \
        tuple[Optional[np.ndarray], Optional[np.ndarray],
              Optional[list[ObservationRecord]], Optional[np.ndarray]]:
    """
    Load EKF simulation results from a .npz file and validate the configuration.

    Args:
    - ekf_results_path: Path to the .npz file containing EKF results.
    - config: The current FilterConfig to validate against the loaded config.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - truth: The ground truth trajectory history (or None if loading failed).
        - z_hist: The measurement history (or None if loading failed).
        - all_obs_records: The list of observation records (or None if loading failed).
        - x_hist: The EKF state estimate history (or None if loading failed).
    """
    logger.info("Attempting to load EKF simulation results from %s", ekf_results_path)
    try:
        with np.load(ekf_results_path, allow_pickle=True) as data:
            # Reconstruct FilterConfig from saved attributes
            loaded_config = FilterConfig(
                N=data['config_N'], steps=data['config_steps'], dt=data['config_dt'],
                sig_r=data['config_sig_r'], sig_rdot=data['config_sig_rdot'],
                q_acc_target=data['config_q_acc_target'], seed=data['config_seed'],
                ISL_range_m=data['config_ISL_range_m']
            )

            # Check if loaded config matches current config
            if loaded_config == config:
                logger.info("Successfully loaded EKF simulation results.")
                # Ensure all_obs_records is converted back to a list of dataclasses
                return data['truth'], data['z_hist'], data['all_obs_records'], data['x_hist']

            logger.warning("Loaded EKF config does not match current config. \
                        Rerunning EKF simulation.")
    except (OSError, ValueError) as e:
        logger.error("Corrupt or unreadable EKF results file. \
                     Rerunning EKF simulation. Error: %s", e)
    except KeyError as e:
        logger.error("EKF results file is missing expected key %s. Rerunning EKF simulation.", e)

    return None, None, None, None


def _simulate_ekf_filter(config: FilterConfig, logger: Any) -> \
        tuple[np.ndarray, np.ndarray, list[ObservationRecord], np.ndarray]:
    """
    Simulate the satellite constellation and run the Clustered EKF to generate
    observation records.

    Args:
    - config: FilterConfig object with simulation parameters.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - truth: The ground truth trajectory history.
        - z_hist: The measurement history.
        - all_obs_records: The list of observation records generated by the EKF.
        - x_hist: The EKF state estimate history.
    """
    logger.info("Simulating satellite constellation to get truth")
    truth, z_hist = simulate_truth_and_meas(
        config.N, config.steps, config.dt, config.sig_r, config.sig_rdot, config.seed
    )

    # --- Start of Clustered EKF Implementation ---
    logger.info("Initialising Clustered EKF with cluster size %s", CLUSTER_SIZE)

    # 1. Create clusters of satellite IDs
    all_sat_ids = list(range(config.N))
    clusters = [all_sat_ids[i:i + CLUSTER_SIZE] for i in range(0, config.N, CLUSTER_SIZE)]

    # 2. Create a list of EKF instances, one for each cluster
    cluster_ekfs = _create_cluster_ekfs(config, truth, clusters, logger)

    # 3. Pre-calculate the mapping from (observer, target) to z_hist index
    z_map = _create_z_map(config.N)

    # 4. Main simulation loop
    logger.info("Collecting observation records using Clustered EKF")
    all_obs_records, x_hist = _run_ekf_main_loop(config, clusters, cluster_ekfs,
                                                 z_map, z_hist, logger)
    # --- End of Clustered EKF Implementation ---

    return truth, z_hist, all_obs_records, x_hist


def _create_cluster_ekfs(config: FilterConfig, truth: np.ndarray,
                         clusters: list[list[int]], logger: Any) -> list[JointEKF]:
    """
    Create a list of JointEKF instances, one for each cluster of satellites.

    Args:
    - config: FilterConfig object with simulation parameters.
    - truth: The ground truth trajectory history.
    - clusters: A list of lists, where each inner list contains the satellite IDs for that cluster.
    - logger: Logger object for logging messages.

    Returns:
    - A list of JointEKF instances, one for each cluster.
    """
    cluster_ekfs = []
    for i, cluster_sat_ids in enumerate(clusters):
        cluster_n = len(cluster_sat_ids)
        cluster_config = FilterConfig(
            N=cluster_n, steps=config.steps, dt=config.dt, sig_r=config.sig_r,
            sig_rdot=config.sig_rdot, q_acc_target=config.q_acc_target,
            seed=config.seed + i, # Use different seed for each cluster
            ISL_range_m=config.ISL_range_m
        )
        # Extract initial truth state for this cluster
        initial_state_slices = [truth[0, sat_id*6:(sat_id+1)*6] for sat_id in cluster_sat_ids]
        cluster_truth_0 = np.concatenate(initial_state_slices)

        cluster_ekfs.append(JointEKF(cluster_config, cluster_truth_0))
        logger.info("Initialised EKF for cluster %d with %d satellites: %s",
                    i, cluster_n, cluster_sat_ids)
    return cluster_ekfs


def _create_z_map(n_satellites: int) -> dict[tuple[int, int], slice]:
    """
    Create a mapping from (observer_id, target_id) pairs to slices of the
    z_hist measurement array.

    Args:
    - n_satellites: The total number of satellites in the simulation.

    Returns:
    - A dictionary where keys are (observer_id, target_id) tuples and
      values are slice objects indicating the columns in z_hist corresponding
      to that observation.
    """
    z_map = {}
    z_idx = 0
    for i in range(n_satellites):
        for j in range(n_satellites):
            if i != j:
                z_map[(i, j)] = slice(z_idx, z_idx + 2)
                z_idx += 2
    return z_map


def _run_ekf_main_loop(config: FilterConfig, clusters: list[list[int]],
                       cluster_ekfs: list[Any],
                       z_map: dict, z_hist: np.ndarray,
                       logger: Any) -> tuple[list[ObservationRecord],
                                             np.ndarray]:
    """
    Run the main loop of the EKF simulation, where each cluster EKF processes its
    predictions and updates based on the measurements.

    Args:
    - config: FilterConfig object with simulation parameters.
    - clusters: A list of lists, where each inner list contains the satellite
                IDs for that cluster.
    - cluster_ekfs: A list of EKF instances corresponding to each cluster.
    - z_map: A dictionary mapping (observer_id, target_id) to slices of the
             z_hist array.
    - z_hist: The measurement history array.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - all_obs_records: A list of all ObservationRecord instances generated by
          the EKF.
        - x_hist: The EKF state estimate history array.

    """
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
                    if obs_id_global != tgt_id_global:
                        z_slice = z_map.get((obs_id_global, tgt_id_global))
                        if z_slice:
                            z_k_cluster_list.append(z_hist[k, z_slice])

            if not z_k_cluster_list:
                continue

            # Update step for the cluster
            z_k_cluster = np.concatenate(z_k_cluster_list)
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

    return all_obs_records, x_hist


def _save_ekf_results(ekf_results_path: str, config: FilterConfig, truth: np.ndarray,
                      z_hist: np.ndarray, all_obs_records: list[ObservationRecord],
                      x_hist: np.ndarray, logger: Any) -> None:
    """
    Save the EKF simulation results to a .npz file.

    Args:
    - ekf_results_path: The path to the .npz file where results will be saved.
    - config: The FilterConfig used for the simulation (saved for reproducibility).
    - truth: The ground truth trajectory history.
    - z_hist: The measurement history.
    - all_obs_records: The list of observation records generated by the EKF.
    - x_hist: The EKF state estimate history.
    - logger: Logger object for logging messages.

    Returns:
    - None. The results are saved to the specified file path.
    """
    logger.info("Saving EKF simulation results to %s", ekf_results_path)
    os.makedirs(os.path.dirname(ekf_results_path), exist_ok=True)
    np.savez_compressed(
        ekf_results_path, config_N=config.N, config_steps=config.steps,
        config_dt=config.dt, config_sig_r=config.sig_r, config_sig_rdot=config.sig_rdot,
        config_q_acc_target=config.q_acc_target, config_seed=config.seed,
        config_ISL_range_m=config.ISL_range_m, truth=truth, z_hist=z_hist,
        all_obs_records=np.array(all_obs_records, dtype=object), # Save as object array
        x_hist=x_hist
    )
    logger.info("EKF simulation results saved successfully.")


async def _run_consensus_phase(config: FilterConfig, truth: np.ndarray,
                               all_obs_records: list[ObservationRecord],
                               save_sim_results: bool, sim_results_path: str, logger: Any) -> \
        tuple[DAG, dict, np.ndarray, set[int]]:
    """
    Run the consensus simulation phase where satellite nodes submit transactions to the DAG
    based on the observation records generated by the EKF.

    Args:
    - config: FilterConfig object with simulation parameters.
    - truth: The ground truth trajectory history.
    - all_obs_records: The list of observation records generated by the EKF.
    - save_sim_results: If True, saves the consensus simulation results to sim_results_path.
    - sim_results_path: Path to save consensus simulation results.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - The final DAG object after all transactions have been processed.
        - A dictionary containing the reputation history for each satellite.
        - The ground truth trajectory history.
        - A set of faulty satellite IDs.
    """
    faulty_ids: set[int] = set()
    poise = ConsensusMechanism()
    queue: asyncio.Queue = asyncio.Queue()
    dag = DAG(queue=queue, consensus_mech=poise)
    listen_task = asyncio.create_task(dag.listen())

    try:
        rep_history = await _execute_consensus_loop(
            config, truth, all_obs_records, queue, dag, faulty_ids, logger
        )

        # Save Consensus Simulation results
        if save_sim_results:
            _save_consensus_results(sim_results_path, dag, rep_history, truth, faulty_ids, logger)
    finally:
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass

    return dag, rep_history, truth, faulty_ids


async def _execute_consensus_loop(config: FilterConfig, truth: np.ndarray,
                                  all_obs_records: list[ObservationRecord],
                                  queue: asyncio.Queue, dag: DAG,
                                  faulty_ids: set[int], logger: Any) -> dict[str, list[float]]:
    """
    Execute the main consensus loop where satellite nodes interact based on the EKF
    observation records.

    Args:
    - config: FilterConfig object with simulation parameters.
    - truth: The ground truth trajectory history.
    - all_obs_records: The list of observation records generated by the EKF.
    - queue: The asyncio.Queue used for communication with the DAG.
    - dag: The DAG instance representing the ledger.
    - faulty_ids: A set to keep track of satellite IDs that have exhibited faulty behaviour.
    - logger: Logger object for logging messages.

    Returns:
    - A dictionary containing the reputation history for each satellite,
      where keys are satellite IDs as strings
      and values are lists of reputation values over time.
    """

    # Create one SatelliteNode for each of the N satellites in the simulation.
    unique_ids = sorted(list(range(config.N)))
    satellites: dict[int, SatelliteNode] = {sid: SatelliteNode(node_id=sid,
                                                               queue=queue) \
                                                                for sid in unique_ids}

    # Initialise rep_history with the starting reputation for all satellites.
    rep_history: dict[str, list[float]] = {str(sid): [satellites[sid].reputation] \
                                           for sid in unique_ids}

    # Group observations by step
    obs_by_step: dict[int, list[ObservationRecord]] = {i: [] for i in range(config.steps)}
    for obs in all_obs_records:
        obs_by_step[obs.step].append(obs)

    for k in range(config.steps):
        # Update satellite positions at each step
        for sid, sat in satellites.items():
            sat.update_position(truth[k, sid*6:(sid+1)*6])

        transactions_submitted_this_step = {sid: False for sid in unique_ids}

        # Iterate through satellites to check for ISL opportunities
        for sid, sat in satellites.items():
            await _process_satellite_interactions(
                sid, sat, k, config, satellites, obs_by_step, faulty_ids, dag,
                transactions_submitted_this_step, logger
            )

            # If no transaction submitted, reputation decays towards neutral
            if not transactions_submitted_this_step[sid]:
                sat.reputation = sat.rep_manager.decay(sat.reputation)

        # Record reputation for all satellites at the end of the step
        for sid in unique_ids:
            rep_history[str(sid)].append(satellites[sid].reputation)

    return rep_history


async def _process_satellite_interactions(sid: int, sat: SatelliteNode, k: int,
                                          config: FilterConfig,
                                          satellites: dict[int, SatelliteNode], obs_by_step: dict,
                                          faulty_ids: set[int], dag: DAG,
                                          transactions_submitted_this_step: dict, logger: Any
                                          ) -> None:
    """
    Process the interactions for a single satellite at a given step, including checking for
    ISL opportunities,
    submitting transactions, and synchronizing with the DAG.

    Args:
    - sid: The ID of the satellite being processed.
    - sat: The SatelliteNode instance for the satellite being processed.
    - k: The current step in the simulation.
    - config: FilterConfig object with simulation parameters.
    - satellites: A dictionary of all SatelliteNode instances, keyed by satellite ID.
    - obs_by_step: A dictionary mapping each step to a list of ObservationRecord
      instances for that step.
    - faulty_ids: A set to keep track of satellite IDs that have exhibited faulty behaviour.
    - dag: The DAG instance representing the ledger.
    - transactions_submitted_this_step: A dictionary tracking whether each satellite
      has submitted a transaction this step.
    - logger: Logger object for logging messages.

    Returns:
    - None. The function updates the state of the satellite and interacts with the DAG as needed.
    """
    for other_sid, other_sat in satellites.items():
        if sid == other_sid:
            continue

        if is_in_isl_range(config.ISL_range_m, sat, other_sat):
            # Find the corresponding observation record
            obs_to_submit = next(
                (obs for obs in obs_by_step.get(k, []) \
                 if obs.observer == sid and obs.target == other_sid),
                None
            )

            if obs_to_submit:
                # Inject dishonest behaviour profiles
                apply_network_faults(obs_to_submit, sid, config.N, k, faulty_ids)

                sat.load_sensor_data(obs_to_submit)
                logger.info("Satellite %s: submitting transaction \
                            for witness of %s.", sid, other_sid)
                await sat.submit_transaction(recipient_address=other_sid)
                transactions_submitted_this_step[sid] = True

            # Once observation submitted, synchronise the DAG on the satellite
            sat.sync_data(dag)


def _save_consensus_results(sim_results_path: str, dag: DAG, rep_history: dict[str, list[float]],
                            truth: np.ndarray, faulty_ids: set[int], logger: Any) -> None:
    """
    Save the consensus simulation results to a .npz file.

    Args:
    - sim_results_path: The path to the .npz file where results will be saved.
    - dag: The final DAG object after all transactions have been processed.
    - rep_history: A dictionary containing the reputation history for each satellite.
    - truth: The ground truth trajectory history.
    - faulty_ids: A set of satellite IDs that exhibited faulty behaviour during the simulation.
    - logger: Logger object for logging messages.

    Returns:
    - None. The results are saved to the specified file path.
    """
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
