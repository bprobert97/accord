"""
Live Evaluation of the Offline CQL Model
"""

import asyncio
import numpy as np
import torch
from cql_offline import load_actor
from harvest_dag_dataset import HarvestConfig, inject_bias_simple
from src.satellite_node import SatelliteNode
from src.consensus_mech import ConsensusMechanism
from src.filters.filter_interface import ObservationRecord

async def evaluate_live_cql():
    cfg = HarvestConfig()
    engine = ConsensusMechanism()
    actor = load_actor("machine_learning/cql_offline.pt")

    satellites = {i: SatelliteNode(node_id=i, consensus_mech=engine) for i in range(cfg.num_sats)}
    comp_ids = set(range(cfg.num_compromised))
    cum_bias = np.zeros(2, dtype=np.float32)

    print("Evaluating CQL Model in live consensus network...")

    for k in range(cfg.filter_burn_in_steps + cfg.honest_steps + cfg.attack_steps):
        for sid, sat in satellites.items():
            sat.peers = [s for other, s in satellites.items() if other != sid]

        for sid, sat in satellites.items():
            for peer in sat.peers:
                if k > cfg.filter_burn_in_steps + cfg.honest_steps and sid in comp_ids:
                    state = np.concatenate([cum_bias, [sat.reputation]]).astype(np.float32)
                    state_tensor = torch.as_tensor(state).unsqueeze(0)

                    with torch.no_grad():
                        action = actor(state_tensor).squeeze(0).numpy()

                    cum_bias = np.clip(cum_bias + action * cfg.action_scale_sigma, -cfg.bias_limit_sigma, cfg.bias_limit_sigma)
                    bias_si = cum_bias * np.array([cfg.sig_r_m, cfg.sig_rdot_mps])

                    mock_record = ObservationRecord(step=k, time=k*cfg.dt_s, observer=sid, target=peer.id, nis=2.0, dof=2, r_vector=[1,0,0], v_vector=[0,1,0], range_m=1000.0, observer_r_eci=[0,0,0])
                    biased_record = inject_bias_simple(mock_record, bias_si)

                    sat.load_sensor_data(biased_record)
                    await sat.submit_transaction(peer.id)

    print("Live Evaluation Complete. Check reputation outputs.")

if __name__ == "__main__":
    asyncio.run(evaluate_live_cql())
