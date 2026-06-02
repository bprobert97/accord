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
import asyncio
import time
import argparse
import functools
import traceback
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from src.logger import get_logger
from src.plotting import extract_nis_transactions, plot_mc_nis_boxplot, \
    generate_corner_plot
from accord_demo import run_consensus_demo, DEFAULT_CONFIG

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
DATA_DIR = os.path.join("sim_data", "mc_results")
EKF_DIR = os.path.join(DATA_DIR, "ekf")
SIM_DIR = os.path.join(DATA_DIR, "sim")
MC_RESULTS_PATH = os.path.join(SIM_DIR, f"mc_results_{ISL_RANGE_KM}km.npz")

def calculate_kpis(rep_history: Optional[Dict[str, List[float]]] = None,
                   faulty_ids: Optional[set[int]] = None,
                   steps: Optional[int] = None,
                   honest_matrix: Optional[np.ndarray] = None,
                   faulty_matrix: Optional[np.ndarray] = None,
                   honest_nis: Optional[List[float]] = None,
                   faulty_nis: Optional[List[float]] = None,
                   detection_threshold: float = 0.5,
                   fpr_offset_percent: float = 0.2,
                   logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    Calculate Key Performance Indicators (KPIs) for a single MC simulation run.

    Args:
        rep_history: Dictionary mapping satellite IDs to their reputation history.
        faulty_ids: List of IDs identifying faulty satellites.
        steps: Total number of simulation steps.
        honest_matrix: A 2D NumPy array where each row is the reputation history
                       of an honest node.
        faulty_matrix: A 2D NumPy array where each row is the reputation history
                       of a faulty node.
        honest_nis: List of NIS values for transactions from honest satellites.
        faulty_nis: List of NIS values for transactions from faulty satellites.
        detection_threshold: The reputation value below which a node is considered
                             "detected" as faulty.
        fpr_offset_percent: The fraction of initial steps to ignore when calculating
                            False Positives (to allow for EKF convergence).
        logger: Optional logger to record undetected faulty nodes.

    Returns:
        A dictionary containing KPIs and processed matrices/NIS lists.
    """
    # If matrices aren't provided, convert the raw rep_history dictionary into NumPy matrices
    if honest_matrix is None or faulty_matrix is None:
        if rep_history is None or faulty_ids is None:
            raise ValueError("Must provide either matrices OR rep_history and faulty_ids")

        honest_ids = sorted([int(sid) for sid in rep_history.keys() if int(sid) not in faulty_ids])
        faulty_ids_list = sorted(list(faulty_ids))
        honest_matrix = np.array([rep_history[str(sid)] for sid in honest_ids])
        faulty_matrix = np.array([rep_history[str(sid)] for sid in faulty_ids_list])
    else:
        # If matrices provided, we try to recover IDs if passed
        honest_ids = None
        faulty_ids_list = sorted(list(faulty_ids)) \
            if faulty_ids is not None else []

    # Infer steps from the matrix shape if not explicitly provided
    if steps is None:
        steps = honest_matrix.shape[1]

    ttds = [] # List to store Time to Detection for each faulty node
    undetected_ids = []
    undetected_reps = []
    false_positives = 0
    true_positives = 0
    total_flips = 0

    # Extract final reputations for reporting
    final_honest_reps = honest_matrix[:, -1]
    final_faulty_reps = faulty_matrix[:, -1]

    # Calculate Time to Detection (TTD) and Recall/FNR for faulty and honest nodes
    for i, history in enumerate(faulty_matrix):
        detected_at = next((idx for idx, rep in enumerate(history) \
                            if rep < detection_threshold), None)
        if detected_at is not None:
            ttds.append(detected_at)
            true_positives += 1
        else:
            # Undetected faulty node
            sid = faulty_ids_list[i] if faulty_ids_list is not None else i
            undetected_ids.append(sid)
            undetected_reps.append(history)
            if logger:
                logger.warning("Faulty satellite %s was NOT detected \
                               (final rep: %.4f, rep history: %s)",
                               sid, history[-1], history)

        # Calculate flips (stability)
        diff = np.diff((history < detection_threshold).astype(int))
        total_flips += np.sum(np.abs(diff))

    # Calculate False Positive Rate (FPR) among honest nodes
    fpr_start_step = int(fpr_offset_percent * steps)
    for history in honest_matrix:
        # A false positive occurs if an honest node's reputation ever drops below the threshold
        if any(rep < detection_threshold for rep in history[fpr_start_step:]):
            false_positives += 1

        # Calculate flips for honest nodes too
        diff = np.diff((history[fpr_start_step:] < detection_threshold).astype(int))
        total_flips += np.sum(np.abs(diff))

    # Normalise Metrics
    num_honest = len(honest_matrix)
    num_faulty = len(faulty_matrix)

    def get_nis_stats(nis_list):
        arr = np.array(nis_list) if nis_list else np.array([])
        if len(arr) == 0:
            return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}
        return {
            "min": float(np.min(arr)),
            "q1": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "q3": float(np.percentile(arr, 75)),
            "max": float(np.max(arr))
        }

    honest_nis_stats = get_nis_stats(honest_nis)
    faulty_nis_stats = get_nis_stats(faulty_nis)

    fpr = (false_positives / num_honest) * 100 if num_honest > 0 else 0
    recall = (true_positives / num_faulty) * 100 if num_faulty > 0 else 0
    fnr = 100 - recall
    precision = (true_positives / (true_positives + false_positives)) * 100 \
                if (true_positives + false_positives) > 0 else 0

    avg_ttd = np.mean(ttds) if ttds else None
    worst_ttd = np.max(ttds) if ttds else None

    mean_honest = np.mean(final_honest_reps) if num_honest > 0 else 0
    mean_faulty = np.mean(final_faulty_reps) if num_faulty > 0 else 0

    return {
        "avg_ttd": avg_ttd,
        "worst_ttd": worst_ttd,
        "fpr": fpr,
        "recall": recall,
        "precision": precision,
        "fnr": fnr,
        "final_honest_rep": mean_honest,
        "final_faulty_rep": mean_faulty,
        "honest_spread": np.std(final_honest_reps) if num_honest > 0 else 0,
        "detection_margin": mean_honest - mean_faulty,
        "flips": total_flips,
        "honest_matrix": honest_matrix,
        "faulty_matrix": faulty_matrix,
        "honest_nis_stats": honest_nis_stats,
        "faulty_nis_stats": faulty_nis_stats,
        "undetected_faulty_ids": undetected_ids,
        "undetected_faulty_reps": np.array(undetected_reps),
        "faulty_ids": faulty_ids_list,
        "honest_ids": honest_ids
    }

def recalculate_all_kpis(all_results: List[Optional[Dict[str, Any]]],
                        detection_threshold: float = 0.5,
                        fpr_offset_percent: float = 0.2) -> List[Optional[Dict[str, Any]]]:
    """
    Recalculate KPIs for a set of Monte Carlo results using new detection parameters.

    Args:
        all_results: A list of KPI dictionaries (one per MC run).
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

        # We reuse the matrices and NIS data already stored in the previous results
        new_kpis = calculate_kpis(
            honest_matrix=res["honest_matrix"],
            faulty_matrix=res["faulty_matrix"],
            faulty_ids=res.get("faulty_ids"),
            honest_nis=res.get("honest_nis"),
            faulty_nis=res.get("faulty_nis"),
            detection_threshold=detection_threshold,
            fpr_offset_percent=fpr_offset_percent
        )
        new_results.append(new_kpis)
    return new_results

def run_single_ekf(run_idx: int) -> bool:
    """
    Run the EKF phase for a single Monte Carlo iteration.
    """
    ekf_path = os.path.join(EKF_DIR, f"ekf_run_{run_idx}.npz")
    log_file = os.path.join(EKF_DIR, f"ekf_run_{run_idx}.log")

    if os.path.exists(ekf_path):
        return True

    logger = get_logger(name=f"EKF_{run_idx}", log_file=log_file)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = DEFAULT_CONFIG
    config.seed += run_idx

    logger.info("Starting EKF Generation for Run %d with Seed %d", run_idx, config.seed)
    try:
        loop.run_until_complete(
            run_consensus_demo(config, save_ekf_results=True, load_ekf_results=False,
                               ekf_results_path=ekf_path, clear_logs=True,
                               log_file=log_file, save_sim_results=False,
                               run_consensus=False)
        )
        return True
    except Exception as e: # pylint: disable=broad-exception-caught
        # We catch everything here so one failed MC iteration doesn't
        # crash the whole pool
        print(f"EKF Run {run_idx} failed: {e}")
        traceback.print_exc()
        return False
    finally:
        loop.close()

def run_single_consensus(run_idx: int,
                          threshold: float = 0.5,
                          fpr_offset: float = 0.2) -> Optional[Dict[str, Any]]:
    """
    Run the Consensus phase for a single Monte Carlo iteration.
    """
    ekf_path = os.path.join(EKF_DIR, f"ekf_run_{run_idx}.npz")
    log_file = os.path.join(SIM_DIR, f"sim_run_{run_idx}.log")

    if not os.path.exists(ekf_path):
        print(f"Missing EKF data for run {run_idx}. Skipping.")
        return None

    logger = get_logger(name=f"SIM_{run_idx}", log_file=log_file)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = DEFAULT_CONFIG
    config.seed += run_idx

    logger.info("Starting Consensus Simulation for Run %d", run_idx)

    try:
        dag, rep_history, _, faulty_ids = loop.run_until_complete(
            run_consensus_demo(config, save_ekf_results=False, load_ekf_results=True,
                               ekf_results_path=ekf_path, clear_logs=True,
                               log_file=log_file, save_sim_results=False,
                               run_consensus=True)
        )

        if rep_history is None or dag is None:
            return None

        # Extract NIS data from DAG
        honest_nis = []
        faulty_nis = []
        for tx, tx_data in extract_nis_transactions(dag):
            sid = tx_data.get("observer")
            nis = getattr(tx.metadata, "nis")

            if sid is None or nis is None:
                continue

            if faulty_ids is not None and int(sid) in faulty_ids:
                faulty_nis.append(float(nis))
            else:
                honest_nis.append(float(nis))


        kpis = calculate_kpis(rep_history, faulty_ids, config.steps,
                              honest_nis=honest_nis, faulty_nis=faulty_nis,
                              detection_threshold=threshold, fpr_offset_percent=fpr_offset,
                              logger=logger)
        return kpis
    except Exception as e: # pylint: disable=broad-exception-caught
        # We catch everything here so one failed MC iteration doesn't
        # crash the whole pool
        print(f"Consensus Run {run_idx} failed: {e}")
        traceback.print_exc()
        return None
    finally:
        loop.close()

def plot_undetected_reputations(all_kpis: List[Dict[str, Any]],
                                threshold: float = 0.5,
                                start_step: int = 0) -> None:
    """
    Plot the full reputation history of every undetected faulty satellite across all runs,
    colour-coded by satellite ID.

    Args:
        all_kpis: List of KPI dictionaries from MC runs.
        threshold: The detection threshold used.
        start_step: The step to start plotting from.
    """
    plt.figure(figsize=(12, 7))

    # Identify all unique IDs that went undetected
    unique_ids = sorted(list(set(
        sid for kpi in all_kpis
        for sid in kpi.get("undetected_faulty_ids", [])
    )))

    if not unique_ids:
        plt.text(0.5, 0.5, "No undetected faulty satellites found",
                 ha="center", va="center", transform=plt.gca().transAxes)
        total_lines = 0
    else:
        # Use a colourmap to assign distinct colours to each ID
        cmap = plt.get_cmap("tab20")
        id_to_color = {sid: cmap(i % 20) for i, sid in enumerate(unique_ids)}

        plotted_legend_ids = set()
        total_lines = 0

        for kpi in all_kpis:
            ids = kpi.get("undetected_faulty_ids", [])
            reps = kpi.get("undetected_faulty_reps", [])

            for sid, history in zip(ids, reps):
                color = id_to_color[sid]
                # Only add to legend once per unique satellite ID
                label = f"Sat {sid}" if sid not in plotted_legend_ids else None

                # Plot from start_step onwards
                steps = np.arange(start_step, len(history))
                plt.plot(steps, history[start_step:], color=color,
                alpha=0.6, linewidth=1.5, label=label)
                plotted_legend_ids.add(sid)
                total_lines += 1

        plt.axhline(threshold, color="black", linestyle="--", label=f"Threshold ({threshold})")

        # Adjust legend position and columns based on the number of items
        num_items = len(plotted_legend_ids)
        if num_items > 15:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
            plt.tight_layout(rect=(0, 0, 0.85, 1))
        elif num_items > 0:
            plt.legend(loc='best', fontsize='small')

    plt.xlabel("Step")
    plt.ylabel("Reputation")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join(DATA_DIR, "mc_undetected_reps.png"))
    plt.show()

def plot_mc_results(all_kpis_raw: List[Optional[Dict[str, Any]]],
                    start_step: int = 0) -> None:
    """
    Aggregate results from all Monte Carlo runs and generate summary plots.

    Generates summary plots for reputation spread, KPI distributions, and 
    undetected faulty nodes.

    Args:
        all_kpis_raw: A list of KPI dictionaries from multiple simulation runs.
        start_step: The step to start plotting from.
    """
    # Filter out failed runs
    all_kpis: List[Dict[str, Any]] = [k for k in all_kpis_raw if k is not None]
    if not all_kpis:
        print("No successful runs to plot.")
        return

    # Plot undetected histories first (new addition)
    # We try to infer the threshold from the first result if possible,
    # though it's typically passed via args in main.
    plot_undetected_reputations(all_kpis, start_step=start_step)

    # 1. Aggregate Reputation Histories
    honest_means_list: List[np.ndarray] = []
    faulty_means_list: List[np.ndarray] = []

    for kpi in all_kpis:
        honest_means_list.append(np.mean(kpi["honest_matrix"], axis=0))
        faulty_means_list.append(np.mean(kpi["faulty_matrix"], axis=0))

    all_honest_means = np.array(honest_means_list)
    all_faulty_means = np.array(faulty_means_list)

    # Slice data based on start_step
    all_honest_means = all_honest_means[:, start_step:]
    all_faulty_means = all_faulty_means[:, start_step:]
    steps = np.arange(start_step, start_step + all_honest_means.shape[1])

    plt.figure(figsize=(10, 6))

    # Honest
    h_mean = np.mean(all_honest_means, axis=0)
    h_std = np.std(all_honest_means, axis=0)
    plt.plot(steps, h_mean, color="green", label="Honest (MC Mean)")
    plt.fill_between(steps, h_mean - h_std, h_mean + h_std, color="green",
                     alpha=0.2, label="Honest Pop. 1 Std. Dev. Spread")

    # Faulty
    f_mean = np.mean(all_faulty_means, axis=0)
    f_std = np.std(all_faulty_means, axis=0)
    plt.plot(steps, f_mean, color="red", label="Faulty (MC Mean)")
    plt.fill_between(steps, f_mean - f_std, f_mean + f_std, color="red",
                     alpha=0.2, label="Faulty Pop. 1 Std. Dev. Spread")

    plt.axhline(0.5, color="gray", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Reputation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DATA_DIR, "mc_reputation.png"))
    plt.show()

    # 2. Plot KPI Distributions
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    # TTD Histogram
    ttds = [float(k.get("avg_ttd", 0)) for k in all_kpis if k.get("avg_ttd") is not None]
    if ttds:
        axes[0].hist(ttds, bins=10, color='skyblue', edgecolor='black')
        axes[0].set_title("Time to Detection (Steps)")
        axes[0].axvline(float(np.mean(ttds)), color='red', linestyle='dashed',
                        label=f'Mean: {np.mean(ttds):.1f}')
        axes[0].legend()

    # FPR Histogram
    fprs = [float(k.get("fpr", 0)) for k in all_kpis]
    axes[1].hist(fprs, bins=10, color='salmon', edgecolor='black')
    axes[1].set_title("False Positive Rate (%)")
    axes[1].axvline(float(np.mean(fprs)), color='red', linestyle='dashed',
                    label=f'Mean: {np.mean(fprs):.1f}%')
    axes[1].legend()

    # Recall/Precision Scatter
    recalls = [float(k.get("recall", 0)) for k in all_kpis]
    precisions = [float(k.get("precision", 0)) for k in all_kpis]
    axes[2].scatter(recalls, precisions, color='purple', alpha=0.5)
    axes[2].set_xlabel("Recall (%)")
    axes[2].set_ylabel("Precision (%)")
    axes[2].set_title("Detection Reliability")
    axes[2].set_xlim(-5, 105)
    axes[2].set_ylim(-5, 105)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "mc_kpis.png"))
    plt.show()

    # Print Summary
    print("--- Monte Carlo Summary ---")
    print(f"Total Runs: {len(all_kpis)}")
    print(f"Mean Recall: {np.mean(recalls):.2f}%")
    print(f"Mean Precision: {np.mean(precisions):.2f}%")
    print(f"Mean FPR: {np.mean(fprs):.2f}%")

    if ttds:
        print(f"Mean TTD: {np.mean(ttds):.2f} steps")
        worst_ttds = [float(k.get('worst_ttd', 0)) for k in \
                      all_kpis if k.get('worst_ttd') is not None]
        if worst_ttds:
            print(f"Worst-Case TTD: {np.max(worst_ttds):.2f} steps")

    print(f"Avg Detection Margin: {np.mean([float(k.get('detection_margin', 0)) \
                                            for k in all_kpis]):.4f}")
    print(f"Avg Honest Spread: {np.mean([float(k.get('honest_spread', 0)) \
                                         for k in all_kpis]):.4f}")
    print(f"Avg Stability (Total Flips): {np.mean([float(k.get('flips', 0)) \
                                                   for k in all_kpis]):.2f}")
    print(f"Avg Final Honest Rep: {np.mean([float(k.get('final_honest_rep', 0)) \
                                            for k in all_kpis]):.4f}")
    print(f"Avg Final Faulty Rep: {np.mean([float(k.get('final_faulty_rep', 0)) \
                                            for k in all_kpis]):.4f}")

    # 4. NIS Median Distribution
    plot_mc_nis_boxplot(all_kpis)

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
    args = parser.parse_args()

    os.makedirs(EKF_DIR, exist_ok=True)
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
                RESULTS = recalculate_all_kpis(RESULTS, detection_threshold=args.threshold,
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

        # Phase 1: EKF Generation
        print(f"Phase 1: Generating EKF data for {args.num_runs} runs...")
        runs_to_gen = [i for i in range(args.num_runs)
                       if not os.path.exists(os.path.join(EKF_DIR, f"ekf_run_{i}.npz"))]

        if runs_to_gen:
            with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
                list(executor.map(run_single_ekf, runs_to_gen))
            print(f"EKF generation completed for {len(runs_to_gen)} runs.")
        else:
            print("All EKF data already exists. Skipping Phase 1.")

        # Phase 2: Consensus Simulation
        print(f"Phase 2: Running Consensus simulations for {args.num_runs} runs...")
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            # Use partial to pass threshold and fpr_offset to run_single_consensus
            sim_func = functools.partial(run_single_consensus, threshold=args.threshold,
                                         fpr_offset=args.fpr_offset)
            RESULTS = list(executor.map(sim_func, range(args.num_runs)))

        end_time = time.time()
        print(f"Monte Carlo simulation completed in {end_time - start_time:.2f} seconds.")

        # Save results
        print(f"Saving Monte Carlo results to {MC_RESULTS_PATH}")
        np.savez_compressed(MC_RESULTS_PATH, results=np.array(RESULTS, dtype=object))

    plot_mc_results(RESULTS, start_step=args.start_step)
    generate_corner_plot()
