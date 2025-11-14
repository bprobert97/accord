# pylint: disable=protected-access, too-many-locals, too-many-statements, broad-exception-caught
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
from typing import Optional
import numpy as np
from src.plotting import plot_consensus_cdf_dof, \
    plot_nis_consistency_overall, plot_reputation
from src.consensus_mech import ConsensusMechanism
from src.dag import DAG
from src.filter import FilterConfig, \
    simulate_truth_and_meas, JointEKF, ObservationRecord
from src.logger import get_logger
from src.satellite_node import SatelliteNode

logger = get_logger()

async def run_consensus_demo(config: FilterConfig) -> tuple[Optional[DAG], Optional[dict]]:
    """
    Run a demo of the consensus mechanism with multiple satellite nodes
    submitting transactions to the DAG.

    Args:
    - Optional filter config for running.

    Returns:
    - The final DAG object after all transactions have been processed.
    """

    truth, z_hist = simulate_truth_and_meas(
        config.N, config.steps, config.dt, config.sig_r, config.sig_rdot
    )

    ekf = JointEKF(config, truth[0])

    # First, run the EKF simulation and collect all observation records
    all_obs_records = []
    x_hist = np.zeros((config.steps, config.N * 6))
    for k in range(config.steps):
        ekf.predict()
        obs_records_step = ekf.update(z_hist[k], k)
        all_obs_records.extend(obs_records_step)
        x_hist[k] = ekf.ekf.x

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

    # Group observations by step
    obs_by_step: list[list[ObservationRecord]] = [[] for _ in range(config.steps)]
    for obs in all_obs_records:
        obs_by_step[obs.step].append(obs)

    # Define satellite IDs for special behavior
    perfect_sat_id = 1
    faulty_sat_id = 2
    intermittent_sat_id = 3

    for k in range(config.steps):
        for obs in obs_by_step[k]:
            sid = obs.observer

            # --- Inject special satellite behavior ---
            if sid == perfect_sat_id:
                # Perfect satellite: always has a very low NIS
                obs.nis = 0.01
            elif sid == faulty_sat_id:
                # Faulty satellite: always has a very high NIS
                obs.nis = 50.0
            elif sid == intermittent_sat_id:
                # Intermittently faulty satellite
                if 200 <= k < 400:
                    # Period of faulty behavior
                    obs.nis = 50.0
                elif 600 <= k < 800:
                    # Period of exceptionally good behavior (recovery)
                    # Demonstrates over long term recovery is possible
                    obs.nis = 0.01

            sat = satellites[sid]
            sat.load_sensor_data(obs)
            logger.info("Satellite %s: submitting transaction.", sid)
            await sat.submit_transaction(recipient_address=123)
            rep_history[str(sid)].append(sat.reputation)

    return dag, rep_history


# Run demo
if __name__ == "__main__":
    default_config = FilterConfig(
        N=4,
        steps=1000,
        dt=60.0,
        sig_r=10.0,
        sig_rdot=0.2,
        q_acc_target=1e-5,
        q_acc_obs=1e-5,   # kept for signature compatibility
        seed=42,
    )

    final_dag, rep_hist = asyncio.run(run_consensus_demo(default_config))
    if final_dag:
        #plot_transaction_dag(final_dag)
        plot_consensus_cdf_dof(final_dag)
        plot_nis_consistency_overall(final_dag)
    if rep_hist:
        plot_reputation(rep_hist)
