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
import logging
import os
import asyncio
import time
import argparse
import functools
import traceback
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from src.logger import get_logger
from src.plotting.mc_plotting import plot_mc_results, DATA_DIR
from src.plotting.plotting import generate_corner_plot, extract_nis_data
from accord_demo import run_consensus_demo, DEFAULT_CONFIG, \
    DemoFilePaths, DemoToggles, FilterType

# Limit NumPy to 1 thread per process to prevent over-subscription
# This is needed for parallel processing using ProcessPoolExecutor.
# Libraries like NumPy and SciPy automatically use all available
# CPU cores for matrix operations, so running in parallel without
# setting these values causes CPU over subscription and causes
# huge performance degradation.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

ISL_RANGE_KM = DEFAULT_CONFIG.ISL_range_m // 1000

# --- MC Configuration ---
NUM_RUNS = 40
NUM_PROCESSES = 4

@dataclass
class RunContext:
    """Stores directory and filter settings for a single Monte Carlo run."""
    filter_type: FilterType
    filter_dir: str
    sim_dir: str

@dataclass
class SatellitePopulationData:
    """
    A dataclass to store data about
    the populations of honest and faulty satellites.
    """
    rep_history: Optional[Dict[str, List[float]]] = None
    faulty_ids: Optional[set[int]] = None
    steps: Optional[int] = None
    honest_matrix: Optional[np.ndarray] = None
    faulty_matrix: Optional[np.ndarray] = None
    honest_nis: Optional[List[float]] = None
    faulty_nis: Optional[List[float]] = None

def calculate_kpis(sat_pop_data: SatellitePopulationData,
                   logger: logging.Logger,
                   detection_threshold: float = 0.5,
                   fpr_offset_percent: float = 0.2) -> Dict[str, Any]:
    """
    Calculates KPIs for a single Monte Carlo run based on the reputation history and faulty IDs.

    Args:
    Inside sat_pop_data:
        - rep_history: A dictionary mapping satellite IDs to their reputation history over time.
        - faulty_ids: A set of satellite IDs that were faulty in this run.
        - steps: The total number of steps in the simulation. If None, it will be inferred from
        the matrices.
        - honest_matrix: A 2D NumPy array of shape (num_honest, steps) containing the reputation
        history of honest satellites.
        - faulty_matrix: A 2D NumPy array of shape (num_faulty, steps) containing the reputation
        history of faulty satellites.
        - honest_nis: A list of NIS values for honest satellites
          (optional, for additional analysis).
        - faulty_nis: A list of NIS values for faulty satellites
          (optional, for additional analysis).
    - logger: A logger for logging warnings about undetected faulty satellites.
    - detection_threshold: The reputation threshold below which a satellite is considered
    detected as faulty.
    - fpr_offset_percent: The percentage of initial steps to ignore when calculating false
    positives, to allow for reputation initialization.

    Returns:
    - A dictionary containing calculated KPIs such as average TTD, worst TTD, FPR, recall,
      precision, FNR, final average reputations for honest and faulty satellites, reputation
      spread among honest satellites, detection margin, total flips in detection status, and
      lists of undetected faulty IDs and their reputations.
    """

    sat_pop_data.honest_matrix, sat_pop_data.faulty_matrix, \
        honest_ids, faulty_ids_list = _resolve_kpi_inputs(
            sat_pop_data.rep_history,
            sat_pop_data.faulty_ids,
            sat_pop_data.honest_matrix,
            sat_pop_data.faulty_matrix
        )

    if sat_pop_data.steps is None:
        sat_pop_data.steps = sat_pop_data.honest_matrix.shape[1]

    if faulty_ids_list is None:
        raise ValueError("Faulty IDs are required to calculate detection metrics.")

    metrics = _compute_detection_metrics(
        sat_pop_data, faulty_ids_list,
        detection_threshold, fpr_offset_percent, logger
    )

    num_honest, num_faulty = len(sat_pop_data.honest_matrix), len(sat_pop_data.faulty_matrix)
    final_honest: np.ndarray = sat_pop_data.honest_matrix[:, -1] if num_honest else np.array([])
    final_faulty: np.ndarray = sat_pop_data.faulty_matrix[:, -1] if num_faulty else np.array([])

    return {
        "avg_ttd": np.mean(metrics['ttds']) if metrics['ttds'] else None,
        "worst_ttd": np.max(metrics['ttds']) if metrics['ttds'] else None,
        "fpr": (metrics['false_positives'] / num_honest) * 100 if num_honest > 0 else 0,
        "recall": (metrics['true_positives'] / num_faulty) * 100 if num_faulty > 0 else 0,
        "precision": (metrics['true_positives'] / (metrics['true_positives'] \
                                                   + metrics['false_positives'])) * 100
                     if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0,
        "fnr": 100 - ((metrics['true_positives'] / num_faulty) * 100 if num_faulty > 0 else 0),
        "final_honest_rep": np.mean(final_honest) if num_honest > 0 else 0,
        "final_faulty_rep": np.mean(final_faulty) if num_faulty > 0 else 0,
        "honest_spread": np.std(final_honest) if num_honest > 0 else 0,
        "detection_margin": (np.mean(final_honest) if num_honest else 0) - \
            (np.mean(final_faulty) if num_faulty else 0),
        "flips": metrics['total_flips'],
        "honest_matrix": sat_pop_data.honest_matrix,
        "faulty_matrix": sat_pop_data.faulty_matrix,
        "honest_nis_stats": _get_nis_stats(sat_pop_data.honest_nis),
        "faulty_nis_stats": _get_nis_stats(sat_pop_data.faulty_nis),
        "undetected_faulty_ids": metrics['undetected_ids'],
        "undetected_faulty_reps": np.array(metrics['undetected_reps']),
        "faulty_ids": faulty_ids_list,
        "honest_ids": honest_ids
    }

def _resolve_kpi_inputs(rep_history: Optional[dict[str, list[float]]],
                        faulty_ids: Optional[set[int]],
                        honest_matrix: Optional[np.ndarray],
                        faulty_matrix: Optional[np.ndarray]
                        ) -> tuple[np.ndarray, np.ndarray,
                                   Optional[List[int]], Optional[List[int]]]:
    """
    Resolves the inputs for KPI calculation, allowing for either raw matrices or a
    reputation history dictionary.

    Args:
    - rep_history: A dictionary mapping satellite IDs to their reputation history over time.
    - faulty_ids: A set of satellite IDs that were faulty in this run.
    - honest_matrix: A 2D NumPy array of shape (num_honest, steps) containing the reputation
      history of honest satellites.
    - faulty_matrix: A 2D NumPy array of shape (num_faulty, steps) containing the reputation
      history of faulty satellites.

    Returns:
    - A tuple containing:
        - honest_matrix: A 2D NumPy array of shape (num_honest, steps) for honest satellites.
        - faulty_matrix: A 2D NumPy array of shape (num_faulty, steps) for faulty satellites.
        - honest_ids: An optional list of satellite IDs corresponding to the rows of honest_matrix.
        - faulty_ids_list: An optional list of satellite IDs corresponding to the rows of
          faulty_matrix.
    """
    if honest_matrix is None or faulty_matrix is None:
        if rep_history is None or faulty_ids is None:
            raise ValueError("Must provide either matrices OR rep_history and faulty_ids")
        honest_ids = sorted([int(sid) for sid in rep_history.keys() if int(sid) not in faulty_ids])
        faulty_ids_list = sorted(list(faulty_ids))
        honest_matrix = np.array([rep_history[str(sid)] for sid in honest_ids])
        faulty_matrix = np.array([rep_history[str(sid)] for sid in faulty_ids_list])
    else:
        honest_ids = None
        faulty_ids_list = sorted(list(faulty_ids)) if faulty_ids is not None else []
    return honest_matrix, faulty_matrix, honest_ids, faulty_ids_list

def _compute_detection_metrics(sat_pop_data: SatellitePopulationData,
                               faulty_ids_list: List[int],
                               threshold: float,
                               fpr_offset: float,
                               logger: logging.Logger) -> Dict[str, Any]:
    """
    Computes detection metrics such as TTDs, false positives, true positives, and
    undetected faulty satellites.

    Args:
    Inside sat_pop_data:
        - honest_matrix: A 2D NumPy array of shape (num_honest, steps) containing the reputation
          history of honest satellites.
        - faulty_matrix: A 2D NumPy array of shape (num_faulty, steps) containing the reputation
          history of faulty satellites.
        - steps: The number of time steps in the reputation history.
    - faulty_ids_list: A list of satellite IDs corresponding to the rows of faulty_matrix.
    - threshold: The reputation threshold below which a satellite is considered
                 detected as faulty.
    - fpr_offset: The percentage of initial steps to ignore when calculating false positives.
    - logger: A logger for logging warnings about undetected faulty satellites.

    Returns:
    - A dictionary containing lists of TTDs, undetected faulty IDs and their reputations, counts of
      true positives, false positives, and total flips in detection status.
    """
    if sat_pop_data.faulty_matrix is None or sat_pop_data.honest_matrix is None:
        raise ValueError("You must supply a faulty data matrix and honest "
                         "data matrix to compute metrics.")

    # Bundle all trackers into the return dictionary immediately to save local variables
    metrics: Dict[str, Any] = {
        'ttds': [], 'undetected_ids': [], 'undetected_reps': [],
        'true_positives': 0, 'false_positives': 0, 'total_flips': 0
    }

    for i, history in enumerate(sat_pop_data.faulty_matrix):
        detected_at = next((idx for idx, rep in enumerate(history) if rep < threshold), None)
        if detected_at is not None:
            metrics['ttds'].append(detected_at)
            metrics['true_positives'] += 1
        else:
            sid = faulty_ids_list[i] if faulty_ids_list[i] else i
            metrics['undetected_ids'].append(sid)
            metrics['undetected_reps'].append(history)
            if logger:
                logger.warning("Faulty satellite %s NOT detected (final rep: %.4f)",
                               sid, history[-1])

        metrics['total_flips'] += np.sum(np.abs(np.diff((history < threshold).astype(int))))

    if sat_pop_data.steps is None:
        logger.warning("Simulation steps not provided. Setting to zero.")
        sat_pop_data.steps = 0

    fpr_start = int(fpr_offset * sat_pop_data.steps)
    for history in sat_pop_data.honest_matrix:
        if any(rep < threshold for rep in history[fpr_start:]):
            metrics['false_positives'] += 1

        metrics['total_flips'] += np.sum(np.abs(np.diff((history[fpr_start:] \
                                                         < threshold).astype(int))))

    return metrics

def _get_nis_stats(nis_list: Optional[List[float]]) -> Dict[str, float]:
    """
    Gets summary statistics for a list of NIS values, including min,
    25th percentile (q1), median, 75th percentile (q3), and max.

    Args:
    - nis_list: A list of NIS values to summarize.

    Returns:
    - A dictionary containing the min, q1, median, q3, and max of the NIS values.
      If the list is empty, all values will be returned as 0.
    """
    arr = np.array(nis_list) if nis_list else np.array([])
    if len(arr) == 0:
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}
    return {
        "min": float(np.min(arr)), "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)), "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr))
    }

def recalculate_all_kpis(all_results: List[Optional[Dict[str, Any]]],
                         logger: logging.Logger,
                         detection_threshold: float = 0.5,
                         fpr_offset_percent: float = 0.2
                         ) -> List[Optional[Dict[str, Any]]]:
    """
    Recalculate KPIs for a set of Monte Carlo results using new detection parameters.

    Args:
        all_results: A list of KPI dictionaries (one per MC run).
        logger: A logger for logging warnings about satellite data.
        detection_threshold: The new reputation threshold to apply.
        fpr_offset_percent: The new initialization offset percentage to apply.

    Returns:
        A list of updated KPI dictionaries.
    """
    new_results: list[Optional[dict]] = []
    for res in all_results:
        if res is None:
            new_results.append(None)
            continue

        sat_pop_data = SatellitePopulationData(
            honest_matrix=res["honest_matrix"],
            faulty_matrix=res["faulty_matrix"],
            faulty_ids=res.get("faulty_ids"),
            honest_nis=res.get("honest_nis"),
            faulty_nis=res.get("faulty_nis")
        )

        # We reuse the matrices and NIS data already stored in the previous results
        new_kpis = calculate_kpis(
            sat_pop_data=sat_pop_data,
            detection_threshold=detection_threshold,
            fpr_offset_percent=fpr_offset_percent,
            logger=logger
        )
        new_results.append(new_kpis)
    return new_results

def run_single_filter(run_idx: int, filter_type: FilterType, filter_dir: str) -> bool:
    """
    Run the filter phase for a single Monte Carlo iteration.
    """
    filter_path = os.path.join(filter_dir, f"{filter_type.value}_run_{run_idx}.npz")
    log_file = os.path.join(filter_dir, f"{filter_type.value}_run_{run_idx}.log")

    if os.path.exists(filter_path):
        return True

    logger = get_logger(name=f"{filter_type.value}_{run_idx}", log_file=log_file)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = replace(DEFAULT_CONFIG, seed=DEFAULT_CONFIG.seed + run_idx)
    toggle = DemoToggles(save_sim_results=False, run_consensus=False)
    fpath = DemoFilePaths(
        filter_type=filter_type,
        filter_results_path=filter_path,
        log_file=log_file
    )

    logger.info("Starting %s Generation for Run %d with Seed %d",
                filter_type.value, run_idx, config.seed)
    try:
        loop.run_until_complete(
            run_consensus_demo(config, toggle, fpath)
        )
        return True
    except Exception as e: # pylint: disable=broad-exception-caught
        # We catch everything here so one failed MC iteration doesn't
        # crash the whole pool
        print(f"Filter Run {run_idx} failed: {e}")
        traceback.print_exc()
        return False
    finally:
        loop.close()

def run_single_consensus(run_idx: int,
                         context: RunContext,
                         threshold: float = 0.5,
                         fpr_offset: float = 0.2) -> Optional[Dict[str, Any]]:
    """
    Run the Consensus phase for a single Monte Carlo iteration.

    Args:
    - run_idx: The index of the run in a series of Monte Carlo runs.
    - context: A RunContext object containing filter type and directory settings.
    - threshold: The reputation threshold below which a satellite is considered
                 detected as faulty.
    - fpr_offset: The percentage of initial steps to ignore when calculating false positives.

    Returns:
    - KPIs for the consensus run, if successful. Else None.
    """
    # Use the dynamic variables passed in
    filter_path = os.path.join(context.filter_dir, f"{context.filter_type.value}_run_{run_idx}.npz")
    log_file = os.path.join(context.sim_dir, f"sim_run_{run_idx}.log")

    if not os.path.exists(filter_path):
        print(f"Missing {context.filter_type.value.upper()} data for run {run_idx}. Skipping.")
        return None

    logger = get_logger(name=f"SIM_{run_idx}", log_file=log_file)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = replace(DEFAULT_CONFIG, seed=DEFAULT_CONFIG.seed + run_idx)

    logger.info("Starting Consensus Simulation for Run %d", run_idx)

    try:
        # Inline the toggle and fpath configurations to save local variables
        dag, rep_history, _, faulty_ids = loop.run_until_complete(
            run_consensus_demo(
                config,
                DemoToggles(save_filter_results=False,
                            load_filter_results=True,
                            save_sim_results=False),
                DemoFilePaths(filter_type=context.filter_type,
                              filter_results_path=filter_path,
                              log_file=log_file)
            )
        )

        if rep_history is None or dag is None:
            return None

        # Use the unified extraction function from plotting.py
        honest_nis, faulty_nis = extract_nis_data(dag, faulty_ids, start_index=0)

        # Inline sat_pop_data directly into the calculate_kpis return statement
        return calculate_kpis(
            sat_pop_data=SatellitePopulationData(
                rep_history=rep_history,
                faulty_ids=faulty_ids,
                steps=config.steps,
                honest_nis=honest_nis,
                faulty_nis=faulty_nis
            ),
            detection_threshold=threshold,
            fpr_offset_percent=fpr_offset,
            logger=logger
        )

    except Exception as e: # pylint: disable=broad-exception-caught
        # We catch everything here so one failed MC iteration doesn't
        # crash the whole pool
        print(f"Consensus Run {run_idx} failed: {e}")
        traceback.print_exc()
        return None
    finally:
        loop.close()


if __name__ == "__main__":
    # e.g. python mc_demo.py --recalculate --threshold 0.3 --fpr-offset 0.1
    parser = argparse.ArgumentParser(description="ACCORD Monte Carlo Simulation")
    parser.add_argument("--num-runs", type=int,
                        default=NUM_RUNS, help="Number of MC runs")
    parser.add_argument("--threshold", type=float,
                        default=0.5, help="Detection threshold for KPIs")
    parser.add_argument("--fpr-offset", type=float,
                        default=0.2, help="FPR offset percent (initialisation ignored)")
    parser.add_argument("--start-step", type=int,
                        default=0, help="Step to start plotting from (convergence)")
    parser.add_argument("--recalculate", action="store_true",
                        help="Recalculate KPIs from saved data")
    parser.add_argument("--filter-type", type=str, choices=["ukf", "ekf"],
                        default="ekf", help="Which filter to use (ukf or ekf)")
    args = parser.parse_args()

    # Convert string to Enum
    selected_filter = FilterType(args.filter_type)
    print(f"Selected filter type: {selected_filter.value.upper()}")

    # Dynamic directories
    FILTER_DIR = os.path.join(DATA_DIR, selected_filter.value)
    SIM_DIR = os.path.join(DATA_DIR, f"sim_{selected_filter.value}")
    MC_RESULTS_PATH = os.path.join(SIM_DIR, f"mc_results_{ISL_RANGE_KM}km.npz")

    os.makedirs(FILTER_DIR, exist_ok=True)
    os.makedirs(SIM_DIR, exist_ok=True)

    RESULTS = None
    if os.path.exists(MC_RESULTS_PATH):
        print(f"Attempting to load Monte Carlo results from {MC_RESULTS_PATH}")
        try:
            with np.load(MC_RESULTS_PATH, allow_pickle=True) as data:
                # results was saved as a single object (the list)
                RESULTS = list(data['results'])
                print(f"Successfully loaded {len(RESULTS)} MC runs.")

            # Check if new keys are missing and auto-trigger recalculate if needed
            NEEDS_RECALCULATE = args.recalculate
            if RESULTS and not NEEDS_RECALCULATE:
                sample = next((r for r in RESULTS if r is not None), None)
                if sample and "recall" not in sample:
                    print("New metrics missing from saved data. Auto-recalculating...")
                    NEEDS_RECALCULATE = True

            if NEEDS_RECALCULATE:
                print(f"Calculating KPIs with threshold={args.threshold}, \
                      fpr_offset={args.fpr_offset}")
                RESULTS = recalculate_all_kpis(RESULTS,
                                               logger=get_logger(),
                                               detection_threshold=args.threshold,
                                               fpr_offset_percent=args.fpr_offset)
                # Save the updated KPIs back to the file
                print(f"Updating saved results at {MC_RESULTS_PATH}")
                np.savez_compressed(MC_RESULTS_PATH, results=np.array(RESULTS, dtype=object))
        except (OSError, ValueError) as e:
            # Catches corrupted files, permission errors, or invalid NumPy archives
            print(f"Corrupt or unreadable MC results file: {e}. Rerunning simulation.")
        except KeyError as e:
            # Catches older save files that don't have the 'results' array
            print(f"MC results file missing expected data: {e}. Rerunning simulation.")

    if RESULTS is None:
        start_time = time.time()

        # Phase 1: Filter Generation
        print(f"Phase 1: Generating {selected_filter.value.upper()} \
              data for {args.num_runs} runs...")
        runs_to_gen = [i for i in range(args.num_runs)
                       if not os.path.exists(os.path.join(FILTER_DIR,
                                                          f"{selected_filter.value}_run_{i}.npz"))]

        if runs_to_gen:
            # Bind the filter arguments just like we do in Phase 2
            filter_func = functools.partial(run_single_filter,
                                            filter_type=selected_filter,
                                            filter_dir=FILTER_DIR)
            with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                list(executor.map(filter_func, runs_to_gen))
            print(f"{selected_filter.value.upper()} generation completed for \
                  {len(runs_to_gen)} runs.")
        else:
            print(f"All {selected_filter.value.upper()} data already exists. Skipping Phase 1.")

        # Phase 2: Consensus Simulation
        print(f"Phase 2: Running Consensus simulations for {args.num_runs} runs...")
        run_context = RunContext(filter_type=selected_filter,
                                filter_dir=FILTER_DIR,
                                sim_dir=SIM_DIR)
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            # Add filter_type, filter_dir, and sim_dir to the partial execution
            sim_func = functools.partial(run_single_consensus,
                                         context=run_context,
                                         threshold=args.threshold,
                                         fpr_offset=args.fpr_offset)
            RESULTS = list(executor.map(sim_func, range(args.num_runs)))

        end_time = time.time()
        print(f"Monte Carlo simulation completed in {end_time - start_time:.2f} seconds.")

        # Save results
        print(f"Saving Monte Carlo results to {MC_RESULTS_PATH}")
        np.savez_compressed(MC_RESULTS_PATH, results=np.array(RESULTS, dtype=object))

    plot_mc_results(RESULTS, start_step=args.start_step)
    generate_corner_plot()
