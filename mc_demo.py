# pylint: disable=protected-access, too-many-locals, too-many-statements, too-many-arguments, too-many-positional-arguments, broad-exception-caught
"""
Monte Carlo Simulation for the ACCORD framework.
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


# --- MC Configuration ---
NUM_RUNS = 40
NUM_PROCESSES = 4
DATA_DIR = "sim_data\\mc_results"
MC_RESULTS_PATH = os.path.join(DATA_DIR, "mc_results.npz")

def calculate_kpis(rep_history: Optional[Dict[str, List[float]]] = None,
                   faulty_ids: Optional[set[int]] = None,
                   steps: Optional[int] = None,
                   honest_matrix: Optional[np.ndarray] = None,
                   faulty_matrix: Optional[np.ndarray] = None,
                   detection_threshold: float = 0.4,
                   fpr_offset_percent: float = 0.2) -> Dict[str, Any]:
    """
    Calculate Key Performance Indicators (KPIs) for a single MC simulation run.

    This function can be initialised in two ways:
    1. By providing raw simulation output (rep_history, faulty_ids, steps).
    2. By providing pre-processed reputation matrices (honest_matrix, faulty_matrix).

    Args:
        rep_history: Dictionary mapping satellite IDs to their reputation history.
        faulty_ids: List of IDs identifying faulty satellites.
        steps: Total number of simulation steps.
        honest_matrix: A 2D NumPy array where each row is the reputation history
                       of an honest node.
        faulty_matrix: A 2D NumPy array where each row is the reputation history
                       of a faulty node.
        detection_threshold: The reputation value below which a node is considered
                             "detected" as faulty.
        fpr_offset_percent: The fraction of initial steps to ignore when calculating
                            False Positives (to allow for EKF convergence).

    Returns:
        A dictionary containing:
            - "avg_ttd": Average Time to Detection for faulty nodes (in steps),
                         or None if none detected.
            - "worst_ttd": Maximum Time to Detection for any faulty node.
            - "fpr": False Positive Rate (%) among honest nodes.
            - "recall": Recall (Sensitivity) - percentage of faulty nodes detected.
            - "precision": Precision - percentage of detected nodes that are actually faulty.
            - "fnr": False Negative Rate (%) - percentage of faulty nodes not detected.
            - "final_honest_rep": Mean reputation of honest nodes at the final step.
            - "final_faulty_rep": Mean reputation of faulty nodes at the final step.
            - "honest_spread": Standard deviation of honest node reputations at the final step.
            - "detection_margin": The gap between honest and faulty final reputations.
            - "flips": Total number of threshold crossings (stability indicator).
            - "honest_matrix": The processed honest reputation matrix.
            - "faulty_matrix": The processed faulty reputation matrix.
    """
    # If matrices aren't provided, convert the raw rep_history dictionary into NumPy matrices
    if honest_matrix is None or faulty_matrix is None:
        if rep_history is None or faulty_ids is None:
            raise ValueError("Must provide either matrices OR rep_history and faulty_ids")

        honest_ids = [int(sid) for sid in rep_history.keys() if int(sid) not in faulty_ids]
        honest_matrix = np.array([rep_history[str(sid)] for sid in honest_ids])
        faulty_matrix = np.array([rep_history[str(sid)] for sid in faulty_ids])

    # Infer steps from the matrix shape if not explicitly provided
    if steps is None:
        steps = honest_matrix.shape[1]

    ttds = [] # List to store Time to Detection for each faulty node
    false_positives = 0
    true_positives = 0
    total_flips = 0

    # Extract final reputations for reporting
    final_honest_reps = honest_matrix[:, -1]
    final_faulty_reps = faulty_matrix[:, -1]

    # Calculate Time to Detection (TTD) and Recall/FNR for faulty nodes
    # TODO what about for after initialisation
    for history in faulty_matrix:
        detected_at = next((i for i, rep in enumerate(history) if rep < detection_threshold), None)
        if detected_at is not None:
            ttds.append(detected_at)
            true_positives += 1

        # Calculate flips (stability)
        # TODO what about for after initialisation
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
        "faulty_matrix": faulty_matrix
    }

def recalculate_all_kpis(all_results: List[Optional[Dict[str, Any]]],
                        detection_threshold: float = 0.4,
                        fpr_offset_percent: float = 0.2) -> List[Optional[Dict[str, Any]]]:
    """
    Recalculate KPIs for a set of Monte Carlo results using new detection parameters.

    This function iterates through previously saved simulation data and reapplies
    the KPI logic without needing to re-run the expensive physics/consensus simulations.

    Args:
        all_results: A list of KPI dictionaries (one per MC run) as returned by calculate_kpis.
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

        # We reuse the matrices already stored in the previous results
        new_kpis = calculate_kpis(
            honest_matrix=res["honest_matrix"],
            faulty_matrix=res["faulty_matrix"],
            detection_threshold=detection_threshold,
            fpr_offset_percent=fpr_offset_percent
        )
        new_results.append(new_kpis)
    return new_results

def run_single_simulation(run_idx: int,
                          threshold: float = 0.4,
                          fpr_offset: float = 0.2) -> Optional[Dict[str, Any]]:
    """
    Wrapper to run a single simulation iteration within a subprocess.

    This function sets up a unique event loop and logger for the simulation run,
    executes the consensus demo, and calculates KPIs for the result.

    Args:
        run_idx: Index of the current Monte Carlo run (used for logging and seeding).
        threshold: Reputation threshold for detection and false positives.
        fpr_offset: Fraction of initial steps to ignore for FPR calculation.

    Returns:
        A dictionary of KPIs if the simulation was successful, otherwise None.
    """
    # Create a unique log file for this run
    log_file = os.path.join(DATA_DIR, f"run_{run_idx}.log")

    # Initialise logger for this process with the unique log file
    # We use the same name "ACCORD" so that all modules using get_logger()
    # will get this redirected logger in this subprocess.
    logger = get_logger(name="ACCORD", log_file=log_file)

    # Create a fresh event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = DEFAULT_CONFIG
    config.seed += run_idx

    logger.info("Starting Monte Carlo Run %d with Seed %d", run_idx, config.seed)

    try:
        # Run the simulation
        # Note: we disable saving/loading EKF results to ensure each MC run is independent
        # and we pass clear_logs=False to avoid clearing other runs' logs
        _, rep_history, _, faulty_ids = loop.run_until_complete(
            run_consensus_demo(config, save_ekf_results=False, load_ekf_results=False,
                               clear_logs=False, log_file=log_file, save_sim_results=False)
        )

        if rep_history is None:
            return None

        kpis = calculate_kpis(rep_history, faulty_ids, config.steps,
                              detection_threshold=threshold, fpr_offset_percent=fpr_offset)
        return kpis
    except Exception as e:
        print(f"Run {run_idx} failed: {e}")
        traceback.print_exc()
        return None
    finally:
        loop.close()

def plot_mc_results(all_kpis_raw: List[Optional[Dict[str, Any]]]) -> None:
    """
    Aggregate results from all Monte Carlo runs and generate summary plots.

    Generates three plots:
    1. Reputation history over time with 1 Std. Dev. spread for honest vs. faulty nodes.
    2. Histograms of Time to Detection (TTD) and False Positive Rate (FPR).
    3. Metrics summary for Recall, Precision, and Stability (Flips).

    Args:
        all_kpis_raw: A list of KPI dictionaries from multiple simulation runs.
    """
    # Filter out failed runs
    all_kpis: List[Dict[str, Any]] = [k for k in all_kpis_raw if k is not None]
    if not all_kpis:
        print("No successful runs to plot.")
        return

    # 1. Aggregate Reputation Histories
    honest_means_list: List[np.ndarray] = []
    faulty_means_list: List[np.ndarray] = []

    for kpi in all_kpis:
        honest_means_list.append(np.mean(kpi["honest_matrix"], axis=0))
        faulty_means_list.append(np.mean(kpi["faulty_matrix"], axis=0))

    all_honest_means = np.array(honest_means_list)
    all_faulty_means = np.array(faulty_means_list)

    steps = np.arange(all_honest_means.shape[1])

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
    plt.title(f"Monte Carlo Reputation Trends ({len(all_kpis)} runs)")
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

if __name__ == "__main__":
    # e.g. python mc_demo.py --recalculate --threshold 0.3 --fpr-offset 0.1
    parser = argparse.ArgumentParser(description="ACCORD Monte Carlo Simulation")
    parser.add_argument("--num-runs", type=int,
                        default=NUM_RUNS, help="Number of MC runs")
    parser.add_argument("--threshold", type=float,
                        default=0.4, help="Detection threshold for KPIs")
    parser.add_argument("--fpr-offset", type=float,
                        default=0.2, help="FPR offset percent (initialisation ignored)")
    parser.add_argument("--recalculate", action="store_true",
                        help="Recalculate KPIs from saved data")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    results = None  # pylint: disable=invalid-name
    if os.path.exists(MC_RESULTS_PATH):
        print(f"Attempting to load Monte Carlo results from {MC_RESULTS_PATH}")
        try:
            with np.load(MC_RESULTS_PATH, allow_pickle=True) as data:
                # results was saved as a single object (the list)
                results = list(data['results'])
                print(f"Successfully loaded {len(results)} MC runs.")

            # Check if new keys are missing and auto-trigger recalculate if needed
            NEEDS_RECALCULATE = args.recalculate
            if results and not NEEDS_RECALCULATE:
                sample = next((r for r in results if r is not None), None)
                if sample and "recall" not in sample:
                    print("New metrics missing from saved data. Auto-recalculating...")
                    NEEDS_RECALCULATE = True

            if NEEDS_RECALCULATE:
                print(f"Calculating KPIs with threshold={args.threshold}, \
                      fpr_offset={args.fpr_offset}")
                results = recalculate_all_kpis(results, detection_threshold=args.threshold,
                                               fpr_offset_percent=args.fpr_offset)
                # Save the updated KPIs back to the file
                print(f"Updating saved results at {MC_RESULTS_PATH}")
                np.savez_compressed(MC_RESULTS_PATH, results=np.array(results, dtype=object))
        except Exception as e:
            print(f"Failed to load MC results: {e}. Rerunning simulation.")

    if results is None:
        start_time = time.time()
        print(f"Starting Monte Carlo simulation with {args.num_runs} \
              runs using {NUM_PROCESSES} processes...")

        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            # Use partial to pass threshold and fpr_offset to run_single_simulation
            sim_func = functools.partial(run_single_simulation, threshold=args.threshold,
                                         fpr_offset=args.fpr_offset)
            results = list(executor.map(sim_func, range(args.num_runs)))

        end_time = time.time()
        print(f"Monte Carlo simulation completed in {end_time - start_time:.2f} seconds.")

        # Save results
        print(f"Saving Monte Carlo results to {MC_RESULTS_PATH}")
        np.savez_compressed(MC_RESULTS_PATH, results=np.array(results, dtype=object))

    plot_mc_results(results)
