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

import argparse
import asyncio
from enum import Enum
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from src.plotting.plotting import  \
    plot_aggregated_reputation, check_consensus_outcomes, \
        plot_nis_boxplot, plot_ground_tracks, \
            calculate_convergence_index, \
                calculate_nis_convergence_index, \
                    calculate_median_percentiles, \
                        plot_constellation, \
                        is_state_evaluated
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG, MockDAG
from src.filters.filter_interface import FilterConfig, \
    simulate_truth_and_meas, ObservationRecord, \
    apply_network_faults
from src.filters.ekf import JointEKF
from src.filters.ukf import JointUKF
from src.filters.ckf import JointCKF
from src.logger import get_logger
from src.satellite_node import SatelliteNode
from src.transaction import Transaction

class FilterType(str, Enum):
    """
    A class to store the filter type as a string.
    """
    CKF = "ckf"
    UKF = "ukf"
    EKF = "ekf"

#------------------
# Constants
CLUSTER_SIZE = 10
DATA_DIR = "sim_data"

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
        (sat1.position.x - sat2.position.x)**2 +
        (sat1.position.y - sat2.position.y)**2 +
        (sat1.position.z - sat2.position.z)**2
    )
    return distance <= isl_range

@dataclass
class StepData:
    """
    A dataclass that stores all of the data for
    one step of the simulation
    """
    k: int
    obs_by_step: dict[int, list[ObservationRecord]]
    tx_this_step: dict[int, bool]

@dataclass
class SimData:
    """
    A dataclass that stores fixed data for
    the entire simulation.
    """
    truth: np.ndarray
    faulty_ids: set[int]
    logger: logging.Logger
    satellites: dict[int, SatelliteNode] = field(default_factory=dict)

@dataclass
class DemoToggles:
    """
    A dataclass for storing toggle values when
    running the consensus demo.
    """
    save_filter_results: bool = True
    load_filter_results: bool = False
    clear_logs: bool = True
    save_sim_results: bool = True
    run_consensus: bool = True

@dataclass
class DemoFilePaths:
    """
    A dataclass for storing file paths
    used in the consensus demo.
    """
    filter_type: FilterType = FilterType.EKF
    filter_results_path: str = ""
    log_file: str = "app.log"
    sim_results_path: str = ""
#------------------
# Main demo function

async def run_consensus_demo(config: FilterConfig,
                             toggles: DemoToggles,
                             file_paths: DemoFilePaths
                             ) -> tuple[Optional[dict[int, DAG]],
                                        Optional[dict[str, list[float]]],
                                        Optional[np.ndarray],
                                        Optional[set[int]]]:
    """
    Run a demo of the consensus mechanism with multiple satellite nodes
    submitting transactions to the DAG.

    Args:
    - config: FilterConfig object with simulation parameters.
    - toggles: An instance of DemoToggles containing the toggled values for running
               the consensus demo.
    - file_paths: An instance of DemoFilePaths containing the desired file paths
                  for running the consensus demo.

    Returns:
    - A tuple containing:
        - The final DAG object after all transactions have been processed.
        - A dictionary containing the reputation history for each satellite.
        - The ground truth trajectory history.
        - A set of faulty satellite IDs.
    """
    logger = get_logger(log_file=file_paths.log_file)
    if toggles.clear_logs:
        clear_log(file_paths.log_file)

    # Attempt to load or generate filter data
    truth, _, all_obs_records, x_hist = _resolve_filter_phase(
        config,
        file_paths,
        toggles.load_filter_results,
        toggles.save_filter_results,
        logger
    )

    # Early return if only filter was requested
    if not toggles.run_consensus:
        logger.info("run_consensus is False. Returning early after filter phase.")
        return None, None, truth, None

    # Ensure data is available for the consensus part
    if all_obs_records is None or x_hist is None or truth is None:
        logger.error("Filter simulation data is not available after loading or running. Exiting.")
        return None, None, None, None

    return await _run_consensus_phase(
        config, truth, all_obs_records,
        toggles.save_sim_results,
        file_paths.sim_results_path
    )


def _resolve_filter_phase(config: FilterConfig,
                          path_toggles: DemoFilePaths,
                          load_filter_results: bool,
                          save_filter_results: bool,
                          logger: logging.Logger) -> \
        tuple[Optional[np.ndarray], Optional[np.ndarray],
              Optional[list[ObservationRecord]], Optional[np.ndarray]]:

    """
    Resolves the filter phase by either loading results from a file or running the simulation.

    Args:
    - config: FilterConfig object with simulation parameters.
    - path_toggles: An instance of DemoFilePaths containing the desired file paths.
    - load_filter_results: If True, attempts to load filter results from filter_results_path.
    - save_filter_results: If True, saves filter results to filter_results_path after
    running simulation.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - truth: The ground truth trajectory history (or None if loading failed).
        - z_hist: The measurement history (or None if loading failed).
        - all_obs_records: The list of observation records (or None if loading failed).
        - x_hist: The filter state estimate history (or None if loading failed).
    """

    truth, z_hist, all_obs_records, x_hist = None, None, None, None

    # Attempt to load filter results if requested
    if load_filter_results and os.path.exists(path_toggles.filter_results_path):
        truth, z_hist, \
            all_obs_records, x_hist = _load_filter_results(path_toggles.filter_results_path,
                                                           config, logger)

    # If filter results were not loaded or loading failed, run the filter simulation
    if truth is None or z_hist is None or all_obs_records is None or x_hist is None:
        truth, z_hist, all_obs_records, x_hist = _simulate_filter(path_toggles.filter_type,
                                                                  config, logger)

        # Save filter results if requested
        if save_filter_results:
            _save_filter_results(path_toggles.filter_results_path, config,
                              (truth, z_hist, all_obs_records, x_hist))

    return truth, z_hist, all_obs_records, x_hist


def _load_filter_results(filter_results_path: str,
                      config: FilterConfig,
                      logger: logging.Logger) -> \
        tuple[Optional[np.ndarray], Optional[np.ndarray],
              Optional[list[ObservationRecord]], Optional[np.ndarray]]:
    """
    Load filter simulation results from a .npz file and validate the configuration.

    Args:
    - filter_results_path: Path to the .npz file containing filter results.
    - config: The current FilterConfig to validate against the loaded config.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - truth: The ground truth trajectory history (or None if loading failed).
        - z_hist: The measurement history (or None if loading failed).
        - all_obs_records: The list of observation records (or None if loading failed).
        - x_hist: The filter state estimate history (or None if loading failed).
    """
    logger.info("Attempting to load filter simulation results from %s", filter_results_path)
    try:
        with np.load(filter_results_path, allow_pickle=True) as data:
            # Reconstruct FilterConfig from saved attributes
            loaded_config = FilterConfig(
                N=data['config_N'], steps=data['config_steps'], dt=data['config_dt'],
                sig_r=data['config_sig_r'], sig_rdot=data['config_sig_rdot'],
                q_acc_target=data['config_q_acc_target'], seed=data['config_seed'],
                ISL_range_m=data['config_ISL_range_m']
            )

            # Check if loaded config matches current config
            if loaded_config == config:
                logger.info("Successfully loaded filter simulation results.")
                # Ensure all_obs_records is converted back to a list of dataclasses
                return data['truth'], data['z_hist'], data['all_obs_records'], data['x_hist']
            logger.warning("Loaded filter config does not match current config." \
            "Loaded config: %s, config: %s Rerunning filter simulation.", loaded_config, config)
    except (OSError, ValueError) as e:
        logger.error("Corrupt or unreadable filter results file. \
                     Rerunning filter simulation. Error: %s", e)
    except KeyError as e:
        logger.error("Filter results file is missing expected key %s. \
                     Rerunning filter simulation.", e)

    return None, None, None, None


def _cluster_by_orbital_physics(truth_0: np.ndarray,
                                N: int,
                                cluster_size: int
                                ) -> list[list[int]]:
    """
    Groups random satellites into filter clusters based on their orbital planes.
    Calculates the angular momentum vector (h = r x v) for each satellite.
    Satellites with similar angular momentum vectors will organically stay
    within physical line-of-sight of each other during the simulation.

    Args:
    - truth_0 (np.ndarray): The initial stacked true state vectors (position and
                            velocity) for all satellites at time step 0.
    - N (int): The total number of satellites in the constellation.
    - cluster_size (int): The desired maximum number of satellites per filter cluster.

    Returns:
    - list[list[int]]: A list of generated satellite clusters. Each inner list
                       represents a single filter cluster and contains the integer IDs
                       of the assigned satellites.
    """
    sat_data = []
    for i in range(N):
        h = np.cross(truth_0[i*6 : i*6+3], truth_0[i*6+3 : i*6+6])
        # Normalise the vector to focus purely on the plane's orientation
        h_norm = h / np.linalg.norm(h)
        sat_data.append((i, h_norm))

    unassigned = set(range(N))
    clusters = []

    # Greedy grouping: Pick a seed satellite, find the (cluster_size - 1)
    # closest matches based on their orbital plane alignment (dot product)
    while unassigned:
        if len(unassigned) <= cluster_size:
            clusters.append(list(unassigned))
            break

        seed_id = unassigned.pop()
        seed_h = next(h for sid, h in sat_data if sid == seed_id)

        alignments = []
        for sid in unassigned:
            h = next(h for s, h in sat_data if s == sid)
            alignments.append((sid, np.dot(seed_h, h)))

        # Sort by dot product approaching 1.0 (highly aligned orbital planes)
        alignments.sort(key=lambda x: x[1], reverse=True)

        current_cluster = [seed_id]
        for i in range(cluster_size - 1):
            if i < len(alignments):
                best_peer_id = alignments[i][0]
                current_cluster.append(best_peer_id)
                unassigned.remove(best_peer_id)

        clusters.append(current_cluster)

    return clusters


def _simulate_filter(filter_type: FilterType,
                     config: FilterConfig,
                     logger: logging.Logger
                     ) -> tuple[np.ndarray,
                                np.ndarray,
                                list[ObservationRecord],
                                np.ndarray]:
    """
    Simulate the satellite constellation and run the Clustered filter to generate
    observation records.

    Args:
    - filter_type: The type of filter to run.
    - config: FilterConfig object with simulation parameters.
    - logger: Logger object for logging messages.

    Returns:
    - A tuple containing:
        - truth: The ground truth trajectory history.
        - z_hist: The measurement history.
        - all_obs_records: The list of observation records generated by the filter.
        - x_hist: The filter state estimate history.
    """
    logger.info("Simulating satellite constellation to get truth")
    truth, z_hist = simulate_truth_and_meas(config, config.walker_delta)

    # --- Start of Clustered filter Implementation ---
    logger.info("Initialising Clustered filter with cluster size %s", CLUSTER_SIZE)

    # 1. Cluster by Orbital Physics
    clusters = _cluster_by_orbital_physics(truth[0], config.N, CLUSTER_SIZE)

    # 2. Create a list of filter instances, one for each cluster
    cluster_filters = _create_cluster_filters(filter_type, config, truth, clusters, logger)

    # 3. Pre-calculate the mapping from (observer, target) to z_hist index
    z_map = _create_z_map(config.N)

    # 4. Main simulation loop
    logger.info("Collecting observation records using Clustered filter")
    all_obs_records, x_hist = _run_filter_main_loop(config, clusters, cluster_filters,
                                                 z_map, z_hist)
    # --- End of Clustered filter Implementation ---

    return truth, z_hist, all_obs_records, x_hist


def _create_cluster_filters(filter_type: FilterType,
                            config: FilterConfig, truth: np.ndarray,
                            clusters: list[list[int]], logger: Any) -> list[Any]:
    """
    Factory function to create the requested filter instances for each cluster of satellites.

    Args:
    - filter_type: The type of filter to create (e.g., CKF, UKF or EKF).
    - config: FilterConfig object with simulation parameters.
    - truth: The ground truth trajectory history.
    - clusters: A list of lists, where each inner list contains the satellite IDs for that cluster.
    - logger: Logger object for logging messages.

    Returns:
    - A list of filter instances, one for each cluster.
    """
    cluster_filters: list[JointEKF | JointUKF | JointCKF | None] = []
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

        if filter_type == FilterType.UKF:
            cluster_filters.append(JointUKF(cluster_config, cluster_truth_0))
        elif filter_type == FilterType.EKF:
            cluster_filters.append(JointEKF(cluster_config, cluster_truth_0))
        elif filter_type == FilterType.CKF:
            cluster_filters.append(JointCKF(cluster_config, cluster_truth_0))
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        logger.info("Initialised %s for cluster %d with %d satellites: %s",
                    filter_type.upper(), i, cluster_n, cluster_sat_ids)
    return cluster_filters


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


def _run_filter_main_loop(config: FilterConfig,
                       clusters: list[list[int]],
                       cluster_filters: list[Any],
                       z_map: dict, z_hist: np.ndarray
                       ) -> tuple[list[ObservationRecord],
                                  np.ndarray]:
    """
    Run the main loop of the filter simulation, where each cluster filter processes its
    predictions and updates based on the measurements.

    Args:
    - config: FilterConfig object with simulation parameters.
    - clusters: A list of lists, where each inner list contains the satellite
                IDs for that cluster.
    - cluster_filters: A list of filter instances corresponding to each cluster.
    - z_map: A dictionary mapping (observer_id, target_id) to slices of the
             z_hist array.
    - z_hist: The measurement history array.

    Returns:
    - A tuple containing:
        - all_obs_records: A list of all ObservationRecord instances generated by
          the filter.
        - x_hist: The filter state estimate history array.
    """
    all_obs_records = []
    x_hist = np.zeros((config.steps, config.N * 6))
    logger = get_logger()

    for k in range(config.steps):
        for cluster_sat_ids, current_filter in zip(clusters, cluster_filters):

            # Delegate the heavy processing to the helper function
            records = _process_cluster_step(
                cluster_sat_ids=cluster_sat_ids,
                current_filter=current_filter,
                z_map=z_map,
                histories = (x_hist, z_hist),
                k=k
            )
            all_obs_records.extend(records)

        logger.info("Completed filter step %d/%d", k + 1, config.steps)

    return all_obs_records, x_hist


def _process_cluster_step(cluster_sat_ids: list[int],
                          current_filter: Any,
                          z_map: dict,
                          histories: tuple[np.ndarray, np.ndarray],
                          k: int) -> list[ObservationRecord]:
    """
    Helper function to predict, update, and log a single cluster for a given timestep.

    Args:
    - cluster_sat_ids: A list of lists, where each inner list contains the satellite
                IDs for that cluster.
    - current_filter: The filter instance for the cluster.
    - z_map: A dictionary mapping (observer_id, target_id) to slices of the
             z_hist array.
    - histories: A tuple of x_hist and z_hist
    - k: The current simulation step

    Returns:
    - A list of all ObservationRecord instances generated by the filter for this step.
    """
    # Unpack the tuple
    x_hist, z_hist = histories

    # Predict step for the cluster
    current_filter.predict()

    # Build the measurement vector `z_k_cluster` for this cluster
    z_k_cluster_list = _get_cluster_measurements(cluster_sat_ids, z_map, z_hist, k)

    if not z_k_cluster_list:
        return []

    # Update step for the cluster (inlining the concatenation)
    obs_records_step = current_filter.update(np.concatenate(z_k_cluster_list), k)

    # Remap local satellite IDs in records to global IDs and store
    for record in obs_records_step:
        record.observer = cluster_sat_ids[record.observer]
        record.target = cluster_sat_ids[record.target]

    # Update the global state history `x_hist`
    for sat_idx_local, sat_idx_global in enumerate(cluster_sat_ids):

        # Dynamically fetch the state array depending on the filter type
        if hasattr(current_filter, 'ukf'):
            local_x = current_filter.ukf.x
        elif hasattr(current_filter, 'ekf'):
            local_x = current_filter.ekf.x
        elif hasattr(current_filter, 'ckf'):
            local_x = current_filter.ckf.x
        else:
            raise ValueError("Filter instance must have either 'ukf' or 'ekf' or 'ckf' attribute.")

        # Inline the slice definitions directly into the array access
        x_hist[k, sat_idx_global * 6 : (sat_idx_global + 1) * 6] = \
            local_x[sat_idx_local * 6 : (sat_idx_local + 1) * 6]

    return obs_records_step

def _get_cluster_measurements(cluster_sat_ids: list[int], z_map: dict,
                              z_hist: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Helper to extract the measurement vectors for a specific cluster.
    Args:
    - cluster_sat_ids: List of global satellite IDs in the cluster.
    - z_map: A dictionary mapping (observer_id, target_id) to slices of the
             z_hist array.
    - z_hist: The measurement history array.
    - k: The current time step.

    Returns:
    - A list of measurement vectors (as numpy arrays) for the cluster at step k.
    """
    z_k_cluster_list = []
    for obs_id_global in cluster_sat_ids:
        for tgt_id_global in cluster_sat_ids:
            if obs_id_global != tgt_id_global:
                z_slice = z_map.get((obs_id_global, tgt_id_global))
                if z_slice:
                    z_k_cluster_list.append(z_hist[k, z_slice])
    return z_k_cluster_list


def _save_filter_results(filter_results_path: str,
                      config: FilterConfig,
                      filter_data: tuple[np.ndarray, np.ndarray,
                                      list[ObservationRecord], np.ndarray]
                      ) -> None:
    """
    Save the filter simulation results to a .npz file.

    Args:
    - filter_results_path: The path to the .npz file where results will be saved.
    - config: The FilterConfig used for the simulation (saved for reproducibility).
    - filter_data: A tuple of:
        - truth: The ground truth trajectory history.
        - z_hist: The measurement history.
        - all_obs_records: The list of observation records generated by the filter.
        - x_hist: The filter state estimate history.

    Returns:
    - None. The results are saved to the specified file path.
    """
    # Unpack the tuple
    truth, z_hist, all_obs_records, x_hist = filter_data

    logger = get_logger()
    logger.info("Saving filter simulation results to %s", filter_results_path)
    os.makedirs(os.path.dirname(filter_results_path), exist_ok=True)
    np.savez_compressed(
        filter_results_path, config_N=config.N, config_steps=config.steps,
        config_dt=config.dt, config_sig_r=config.sig_r, config_sig_rdot=config.sig_rdot,
        config_q_acc_target=config.q_acc_target, config_seed=config.seed,
        config_ISL_range_m=config.ISL_range_m, truth=truth, z_hist=z_hist,
        all_obs_records=np.array(all_obs_records, dtype=object), # Save as object array
        x_hist=x_hist
    )
    logger.info("Filter simulation results saved successfully.")


async def _run_consensus_phase(config: FilterConfig,
                               truth: np.ndarray,
                               all_obs_records: list[ObservationRecord],
                               save_sim_results: bool,
                               sim_results_path: str) -> \
        tuple[dict[int, DAG], dict, np.ndarray, set[int]]:
    """
    Run the consensus simulation phase where satellite nodes submit transactions to the DAG
    based on the observation records generated by the filter.

    Args:
    - config: FilterConfig object with simulation parameters.
    - truth: The ground truth trajectory history.
    - all_obs_records: The list of observation records generated by the filter.
    - save_sim_results: If True, saves the consensus simulation results to sim_results_path.
    - sim_results_path: Path to save consensus simulation results.

    Returns:
    - A tuple containing:
        - A dict of DAG objects
        - A dictionary containing the reputation history for each satellite.
        - The ground truth trajectory history.
        - A set of faulty satellite IDs.
    """
    faulty_ids: set[int] = set()
    sim_data = SimData(
        truth=truth,
        faulty_ids=faulty_ids,
        logger=get_logger())

    rep_history = await _execute_consensus_loop(
        config, sim_data, all_obs_records
    )

    # For saving/plotting, we collect all localised DAGs
    # Key: Satellite ID, Value: That satellite's local DAG instance
    nodes_dags = {sid: sat.dag for sid, sat in sim_data.satellites.items()}

    if save_sim_results:
        _save_consensus_results(sim_results_path, sim_data, rep_history, nodes_dags)

    return nodes_dags, rep_history, truth, faulty_ids


async def _execute_consensus_loop(config: FilterConfig,
                                  sim_data: SimData,
                                  all_obs_records: list[ObservationRecord]
                                  ) -> dict[str, list[float]]:
    """
    Execute the main consensus loop where satellite nodes interact based on the filter
    observation records.

    Args:
    - config: FilterConfig object with simulation parameters.
    - sim_data: The fixed simulation data
    - all_obs_records: The list of observation records generated by the filter.

    Returns:
    - A dictionary containing the reputation history for each satellite,
      where keys are satellite IDs as strings
      and values are lists of reputation values over time.
    """

    # Create one SatelliteNode for each of the N satellites in the simulation.
    unique_ids = sorted(list(range(config.N)))

    # Every satellite gets its own fresh node and its own internal consensus processor
    sim_data.satellites = {
        sid: SatelliteNode(node_id=sid, consensus_mech=ConsensusMechanism()) \
        for sid in unique_ids
    }

    rep_history = {str(sid): [sim_data.satellites[sid].reputation] for sid in unique_ids}

    # Group observations by step
    obs_by_step: dict[int, list[ObservationRecord]] = {i: [] for i in range(config.steps)}
    for obs in all_obs_records:
        obs_by_step[obs.step].append(obs)

    for k in range(config.steps):
        # 1. Update positions based on orbital truth data
        for sid, sat in sim_data.satellites.items():
            sat.update_position(sim_data.truth[k, sid*6:(sid+1)*6])

        # 2. Dynamic P2P Topology Discovery
        # Before sending messages, calculate which satellites are in view of each other at step k
        for sid, sat in sim_data.satellites.items():
            sat.peers = [
                other_sat for other_sid, other_sat in sim_data.satellites.items()
                if sid != other_sid and is_in_isl_range(config.ISL_range_m, sat, other_sat)
            ]

        transactions_submitted_this_step = {sid: False for sid in unique_ids}
        step_data = StepData(
            k=k,
            obs_by_step=obs_by_step,
            tx_this_step=transactions_submitted_this_step
        )

        # 3. Process Interactions
        for sid, sat in sim_data.satellites.items():
            await _process_satellite_interactions(
                sid=sid,
                sat=sat,
                step_data=step_data,
                config=config,
                sim_data=sim_data
            )

            if not transactions_submitted_this_step[sid]:
                sat.reputation = sat.rep_manager.decay(sat.reputation)

        for sid in unique_ids:
            rep_history[str(sid)].append(sim_data.satellites[sid].reputation)

    return rep_history


async def _process_satellite_interactions(sid: int,
                                          sat: SatelliteNode,
                                          step_data: StepData,
                                          config: FilterConfig,
                                          sim_data: SimData
                                          ) -> None:
    """
    Process the interactions for a single satellite at a given step, including checking for
    ISL opportunities,
    submitting transactions, and synchronizing with the DAG.

    Args:
    - sid: The ID of the satellite being processed.
    - sat: The SatelliteNode instance for the satellite being processed.
    - step_data: The data for step k of the simulation
    - config: FilterConfig object with simulation parameters.
    - sim_data: Fixed data for the simulation.

    Returns:
    - None. The function updates the state of the satellite and interacts
            with the satellite's DAG as needed.
    """
    # Look through the peers we calculated are currently in range
    for other_sat in sat.peers:
        other_sid = other_sat.id

        # When an ISL link is active, ask our peer for historical
        # data transactions we missed
        #await sat.request_sync_from_peer(other_sat)

        # Find the corresponding observation record
        obs_to_submit = next(
            (obs for obs in step_data.obs_by_step.get(step_data.k, []) \
             if obs.observer == sid and obs.target == other_sid),
            None
        )

        if obs_to_submit:
            # Inject dishonest behaviour profiles
            apply_network_faults(obs_to_submit,
                                 sid, config.N,
                                 step_data.k,
                                 sim_data.faulty_ids)

            sat.load_sensor_data(obs_to_submit)
            sim_data.logger.info("Satellite %s: submitting transaction of target %s\
                                 and broadcasting to peers.", sid, other_sid)

            # This saves locally and calls peer.receive_transaction() across the network
            await sat.submit_transaction(recipient_address=other_sid)
            step_data.tx_this_step[sid] = True


def _save_consensus_results(sim_results_path: str,
                            sim_data: SimData,
                            rep_history: dict[str, list[float]],
                            nodes_dags: dict[int, DAG]
                            ) -> None:
    """
    Save the consensus simulation results to a .npz file.

    Args:
    - sim_results_path: The path to the .npz file where results will be saved.
    - Inside sim_data:
        - dag: The final DAG object after all transactions have been processed.
        - truth: The ground truth trajectory history.
        - faulty_ids: A set of satellite IDs that exhibited faulty behaviour during the simulation.
        - logger: Logger object for logging messages.
    - rep_history: A dictionary containing the reputation history for each satellite.
    - nodes_dags: A dictionary of the local DAGs for each satellite in the simulation.

    Returns:
    - None. The results are saved to the specified file path.
    """
    sim_data.logger.info("Saving Simulation results to %s", sim_results_path)
    os.makedirs(os.path.dirname(sim_results_path), exist_ok=True)

    unified_ledger: dict[str, list[Transaction]] = {}
    unified_states: dict[str, dict] = {}  # Store the network's consensus opinions

    for _, local_dag in nodes_dags.items():
        for tx_hash, tx_list in local_dag.ledger.items():
            current_state = local_dag.local_consensus_states.get(tx_hash, {})

            if tx_hash not in unified_ledger:
                unified_ledger[tx_hash] = tx_list
                unified_states[tx_hash] = current_state
            else:
                existing_state = unified_states.get(tx_hash, {})

                # Replaced duplicate blocks with the clean shared helper evaluation check
                if is_state_evaluated(current_state) and not is_state_evaluated(existing_state):
                    unified_ledger[tx_hash] = tx_list
                    unified_states[tx_hash] = current_state

    np.savez_compressed(
        sim_results_path,
        dag_ledger=unified_ledger,  # type: ignore[arg-type]
        global_consensus_states=unified_states,  # type: ignore[arg-type]
        rep_history=rep_history,  # type: ignore[arg-type]
        truth=sim_data.truth,
        faulty_ids=np.array(list(sim_data.faulty_ids))
    )
    sim_data.logger.info("Simulation results saved successfully with \
                         %d unique evaluated transactions.", len(unified_ledger))

# Run demo
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="ACCORD Filter Demonstration")
    parser.add_argument(
        "--filter-type",
        type=str,
        choices=["ukf", "ekf", "ckf"],
        default="ekf",
        help="Which filter to use (ukf or ekf or ckf)"
    )
    parser.add_argument(
            "--walker-delta",
            type=bool,
            choices=[True, False],
            default=False,
            help="Would you like to generate a Walker Delta constellation \
                instead of a random topology?"
        )
    args = parser.parse_args()

    DEFAULT_CONFIG.walker_delta = args.walker_delta

    accord_logger = get_logger()

    # Define which filter to use based on the command line argument
    SELECTED_FILTER = FilterType(args.filter_type)
    accord_logger.info("Selected filter type: %s", SELECTED_FILTER.value)

    # Dynamically build the paths based on the selected filter
    FILTER_RESULTS_PATH = os.path.join(DATA_DIR, f"{SELECTED_FILTER.value}_simulation_results.npz")
    SIM_RESULTS_PATH = os.path.join(DATA_DIR, f"sim_results_{SELECTED_FILTER.value}.npz")

    FINAL_DAG: dict[int, DAG] | MockDAG | None = None
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

                    # Fetch states, default to empty dict if older sim data file
                    states = simulated_data.get('global_consensus_states', np.array({})).item()
                    if not isinstance(states, dict):
                        states = {}

                    FINAL_DAG = MockDAG(ledger=dag_ledger, local_consensus_states=states)
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
        toggle = DemoToggles(load_filter_results=True)
        # Pass our dynamically generated paths to DemoFilePaths
        paths = DemoFilePaths(
            filter_type=SELECTED_FILTER,
            filter_results_path=FILTER_RESULTS_PATH,
            sim_results_path=SIM_RESULTS_PATH
        )

        FINAL_DAG, REP_HIST, TRUTH, FAULTY_IDS = asyncio.run(
            run_consensus_demo(DEFAULT_CONFIG, toggles=toggle,
                               file_paths=paths))

        # Copy the log file to the sim_data directory
        if os.path.exists("app.log"):
            shutil.copy("app.log", os.path.join(DATA_DIR, "app.log"))
            accord_logger.info("Copied app.log to %s.", DATA_DIR)

    # Use the results for plotting
    if FINAL_DAG is not None and FAULTY_IDS is not None:
        plot_nis_boxplot(FINAL_DAG, compromised_ids=FAULTY_IDS)
        NIS_CONVERGENCE_INDEX = calculate_nis_convergence_index(FINAL_DAG,\
            compromised_ids=FAULTY_IDS)
        plot_nis_boxplot(FINAL_DAG, compromised_ids=FAULTY_IDS, \
            convergence_index=NIS_CONVERGENCE_INDEX)
        calculate_median_percentiles()
        check_consensus_outcomes(FINAL_DAG)
    if REP_HIST and FAULTY_IDS is not None:
        CONVERGENCE_IDX = calculate_convergence_index(REP_HIST, compromised_ids=FAULTY_IDS)
        plot_aggregated_reputation(REP_HIST, compromised_ids=FAULTY_IDS,
                                   start_at_full_constellation=False,
                                   convergence_index=CONVERGENCE_IDX)
    if TRUTH is not None and FAULTY_IDS is not None:
        plot_ground_tracks(TRUTH, DEFAULT_CONFIG.N)
        plot_constellation(TRUTH, DEFAULT_CONFIG.N)
