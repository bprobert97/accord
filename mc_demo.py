# pylint: disable=protected-access, too-many-locals, too-many-statements
"""
Monte Carlo Simulation for the ACCORD framework.
"""

import asyncio
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from src.filter import FilterConfig
from src.logger import get_logger
from accord_demo import run_consensus_demo


# --- MC Configuration ---
NUM_RUNS = 10
NUM_PROCESSES = 4 # Adjust based on your CPU cores
DATA_DIR = "sim_data\\mc_results"
MC_RESULTS_PATH = os.path.join(DATA_DIR, "mc_results.npz")

def calculate_kpis(rep_history, faulty_ids, steps):
    """
    Calculate Key Performance Indicators (KPIs) for a single simulation run.
    """
    threshold = 0.4
    ttds = [] # Time to Detection
    false_positives = 0
    honest_ids = [int(sid) for sid in rep_history.keys() if int(sid) not in faulty_ids]

    # Final Reputations
    final_honest_reps = [rep_history[str(sid)][-1] for sid in honest_ids]
    final_faulty_reps = [rep_history[str(sid)][-1] for sid in faulty_ids]

    # Time to Detection (TTD)
    for sid in faulty_ids:
        history = rep_history[str(sid)]
        detected_at = next((i for i, rep in enumerate(history) if rep < threshold), None)
        if detected_at is not None:
            ttds.append(detected_at)

    # False Positive Rate (FPR) TODO not sure this is right
    for sid in honest_ids:
        history = rep_history[str(sid)]
        if any(rep < threshold for rep in history[int(0.2*steps):]): # Ignore initialization
            false_positives += 1

    fpr = (false_positives / len(honest_ids)) * 100 if honest_ids else 0
    avg_ttd = np.mean(ttds) if ttds else None

    return {
        "avg_ttd": avg_ttd,
        "fpr": fpr,
        "final_honest_rep": np.mean(final_honest_reps) if final_honest_reps else 0,
        "final_faulty_rep": np.mean(final_faulty_reps) if final_faulty_reps else 0,
        "honest_matrix": np.array([rep_history[str(sid)] for sid in honest_ids]),
        "faulty_matrix": np.array([rep_history[str(sid)] for sid in faulty_ids])
    }

def run_single_simulation(run_idx):
    """
    Wrapper to run a single simulation iteration.
    """
    # Create a unique log file for this run
    log_file = os.path.join(DATA_DIR, f"run_{run_idx}.log")

    # Initialize logger for this process with the unique log file
    # We use the same name "ACCORD" so that all modules using get_logger() 
    # will get this redirected logger in this subprocess.
    logger = get_logger(name="ACCORD", log_file=log_file)
    logger.info("Starting Monte Carlo Run %d", run_idx)

    # Create a fresh event loop for this process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = FilterConfig(
        N=50, # Smaller N for faster MC testing
        steps=300,
        dt=60.0,
        sig_r=10.0,
        sig_rdot=0.2,
        q_acc_target=1e-5,
        q_acc_obs=1e-5,
        seed=100 + run_idx,
    )

    try:
        # Run the simulation
        # Note: we disable saving/loading EKF results to ensure each MC run is independent
        # and we pass clear_logs=False to avoid clearing other runs' logs
        _, rep_history, _, faulty_ids = loop.run_until_complete(
            run_consensus_demo(config, save_ekf_results=False, load_ekf_results=False, 
                               clear_logs=True, log_file=log_file, save_sim_results=False)
        )

        if rep_history is None:
            return None

        kpis = calculate_kpis(rep_history, faulty_ids, config.steps)
        return kpis
    except Exception as e:
        print(f"Run {run_idx} failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        loop.close()

def plot_mc_results(all_kpis):
    """
    Plot aggregated results from all MC runs.
    """
    # Filter out failed runs
    all_kpis = [k for k in all_kpis if k is not None]
    if not all_kpis:
        print("No successful runs to plot.")
        return

    # 1. Aggregate Reputation Histories
    # We'll average the honest/faulty averages across runs
    all_honest_means = []
    all_faulty_means = []

    for kpi in all_kpis:
        all_honest_means.append(np.mean(kpi["honest_matrix"], axis=0))
        all_faulty_means.append(np.mean(kpi["faulty_matrix"], axis=0))

    all_honest_means = np.array(all_honest_means)
    all_faulty_means = np.array(all_faulty_means)

    steps = np.arange(all_honest_means.shape[1])

    plt.figure(figsize=(10, 6))

    # Honest
    h_mean = np.mean(all_honest_means, axis=0)
    h_std = np.std(all_honest_means, axis=0)
    plt.plot(steps, h_mean, color="green", label="Honest (MC Mean)")
    plt.fill_between(steps, h_mean - 2*h_std, h_mean + 2*h_std, color="green", alpha=0.2, label="Honest 95% CI")

    # Faulty
    f_mean = np.mean(all_faulty_means, axis=0)
    f_std = np.std(all_faulty_means, axis=0)
    plt.plot(steps, f_mean, color="red", label="Faulty (MC Mean)")
    plt.fill_between(steps, f_mean - 2*f_std, f_mean + 2*f_std, color="red", alpha=0.2, label="Faulty 95% CI")

    plt.axhline(0.5, color="gray", linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Reputation")
    plt.title(f"Monte Carlo Results ({len(all_kpis)} runs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DATA_DIR, "mc_reputation.png"))
    plt.show()

    # 2. Plot KPI Distributions
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    ttds = [k["avg_ttd"] for k in all_kpis if k["avg_ttd"] is not None]
    if ttds:
        axes[0].hist(ttds, bins=10, color='skyblue', edgecolor='black')
        axes[0].set_title("Time to Detection (Steps)")
        axes[0].axvline(np.mean(ttds), color='red', linestyle='dashed', label=f'Mean: {np.mean(ttds):.1f}')
        axes[0].legend()

    fprs = [k["fpr"] for k in all_kpis]
    axes[1].hist(fprs, bins=10, color='salmon', edgecolor='black')
    axes[1].set_title("False Positive Rate (%)")
    axes[1].axvline(np.mean(fprs), color='red', linestyle='dashed', label=f'Mean: {np.mean(fprs):.1f}%')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "mc_kpis.png"))
    plt.show()

    # Print Summary
    print("--- Monte Carlo Summary ---")
    print(f"Total Runs: {len(all_kpis)}")
    if ttds:
        print(f"Mean Time to Detection: {np.mean(ttds):.2f} steps")
    print(f"Mean False Positive Rate: {np.mean(fprs):.2f}%")
    print(f"Avg Final Honest Rep: {np.mean([k['final_honest_rep'] for k in all_kpis]):.4f}")
    print(f"Avg Final Faulty Rep: {np.mean([k['final_faulty_rep'] for k in all_kpis]):.4f}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    results = None
    if os.path.exists(MC_RESULTS_PATH):
        print(f"Attempting to load Monte Carlo results from {MC_RESULTS_PATH}")
        try:
            with np.load(MC_RESULTS_PATH, allow_pickle=True) as data:
                # results was saved as a single object (the list)
                results = data['results'].tolist()
                print(f"Successfully loaded {len(results)} MC runs.")
        except Exception as e:
            print(f"Failed to load MC results: {e}. Rerunning simulation.")

    if results is None:
        start_time = time.time()
        print(f"Starting Monte Carlo simulation with {NUM_RUNS} runs using {NUM_PROCESSES} processes...")

        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            results = list(executor.map(run_single_simulation, range(NUM_RUNS)))

        end_time = time.time()
        print(f"Monte Carlo simulation completed in {end_time - start_time:.2f} seconds.")

        # Save results
        print(f"Saving Monte Carlo results to {MC_RESULTS_PATH}")
        np.savez_compressed(MC_RESULTS_PATH, results=np.array(results, dtype=object))

    plot_mc_results(results)
