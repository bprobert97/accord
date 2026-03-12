import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from graphviz import Digraph
from mc_demo import recalculate_all_kpis

# --- Configuration ---
SIM_RESULTS_PATH = "sim_data/sim_results.npz"
MC_RESULTS_PATH = "sim_data/mc_results/mc_results.npz"

st.set_page_config(page_title="ACCORD Dashboard", layout="wide")

st.title("🛰️ ACCORD: Autonomous Cooperative Consensus Orbit Determination")
st.markdown("""
This dashboard provides interactive visualisation and analysis for the ACCORD framework,
leveraging decentralised reputation and DAG-based ledgers for satellite orbit determination.
""")

# --- Helper Functions ---
@st.cache_data
def load_sim_results(path):
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return {
                "dag_ledger": data["dag_ledger"].item(),
                "rep_history": data["rep_history"].item(),
                "truth": data["truth"],
                "faulty_ids": set(data["faulty_ids"])
            }
    except Exception as e:
        st.error(f"Error loading simulation results: {e}")
        return None

@st.cache_data
def load_mc_results(path):
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return list(data['results'])
    except Exception as e:
        st.error(f"Error loading MC results: {e}")
        return None

# --- Sidebar ---
st.sidebar.header("Data Source")
sim_data = load_sim_results(SIM_RESULTS_PATH)
mc_data = load_mc_results(MC_RESULTS_PATH)

if sim_data:
    st.sidebar.success("Loaded Single Simulation Data")
else:
    st.sidebar.warning("Single Simulation Data (sim_results.npz) not found.")

if mc_data:
    st.sidebar.success(f"Loaded {len(mc_data)} Monte Carlo Runs")
else:
    st.sidebar.warning("MC Data (mc_results.npz) not found.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Results Explorer", "📈 Sensitivity Analysis", "🕸️ DAG Viewer"])

# --- Tab 1: Results Explorer ---
with tab1:
    if sim_data:
        st.header("Simulation Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Reputation History")
            rep_history = sim_data["rep_history"]
            faulty_ids = sim_data["faulty_ids"]

            # Convert to DataFrame for Plotly
            rep_df_list = []
            for sid, history in rep_history.items():
                is_faulty = int(sid) in faulty_ids
                for step, rep in enumerate(history):
                    rep_df_list.append({
                        "Step": step,
                        "Reputation": rep,
                        "Satellite": f"Sat {sid}",
                        "Status": "Faulty" if is_faulty else "Honest"
                    })
            rep_df = pd.DataFrame(rep_df_list)

            fig_rep = px.line(rep_df, x="Step", y="Reputation", color="Satellite", 
                             line_dash="Status", color_discrete_sequence=px.colors.qualitative.Safe)
            fig_rep.add_hline(y=0.5, line_dash="dot", annotation_text="Neutral", line_color="gray")
            st.plotly_chart(fig_rep, use_container_width=True)

        with col2:
            st.subheader("Satellite Ground Tracks")
            # Existing matplotlib plot
            fig_map, ax = plt.subplots(figsize=(10, 6))
            # We need to adapt plot_ground_tracks to take an axis or return a figure
            # For now, let's just use the truth data to show a simple plot if we can't easily hijack the src function
            truth = sim_data["truth"]
            num_sats = int(truth.shape[1] / 6)

            # Use the existing function but capture the figure
            # Note: plot_ground_tracks calls plt.show() which might be tricky in streamlit
            # Let's write a simplified version for now or refactor src/plotting.py later
            for i in range(num_sats):
                pos_hist = truth[:, i*6:i*6+3]
                r = np.linalg.norm(pos_hist, axis=1)
                lat = np.degrees(np.arcsin(np.clip(pos_hist[:, 2] / r, -1, 1)))
                lon = np.degrees(np.arctan2(pos_hist[:, 1], pos_hist[:, 0]))
                ax.scatter(lon[-1], lat[-1], label=f"Sat {i}", s=10)

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_map)
            st.info("Showing current positions. For full ground tracks, see 'accord_demo.py' outputs.")
    else:
        st.info("Please run `python accord_demo.py` to generate simulation data.")

# --- Tab 2: Sensitivity Analysis ---
with tab2:
    if mc_data:
        st.header("Monte Carlo Sensitivity Analysis")
        st.markdown("Recalculate KPIs instantly without re-running simulations.")

        col1, col2 = st.columns([1, 3])

        with col1:
            threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.4, 0.05)
            fpr_offset = st.slider("FPR Offset (Initial % ignored)", 0.0, 0.5, 0.2, 0.05)

            if st.button("Recalculate KPIs"):
                new_results = recalculate_all_kpis(mc_data, detection_threshold=threshold, fpr_offset_percent=fpr_offset)

                # Filter out failed runs
                valid_kpis = [k for k in new_results if k is not None]
                ttds = [k["avg_ttd"] for k in valid_kpis if k["avg_ttd"] is not None]
                fprs = [k["fpr"] for k in valid_kpis]

                st.metric("Mean TTD", f"{np.mean(ttds):.2f} steps" if ttds else "N/A")
                st.metric("Mean FPR", f"{np.mean(fprs):.2f}%")

        with col2:
            # We can't easily get new_results here without state, so let's just do it once
            new_results = recalculate_all_kpis(mc_data, detection_threshold=threshold, fpr_offset_percent=fpr_offset)
            valid_kpis = [k for k in new_results if k is not None]

            all_honest_means = np.array([np.mean(k["honest_matrix"], axis=0) for k in valid_kpis])
            all_faulty_means = np.array([np.mean(k["faulty_matrix"], axis=0) for k in valid_kpis])
            steps = np.arange(all_honest_means.shape[1])

            fig_mc = go.Figure()
            # Honest Mean & CI
            h_mean = np.mean(all_honest_means, axis=0)
            h_std = np.std(all_honest_means, axis=0)
            fig_mc.add_trace(go.Scatter(x=steps, y=h_mean, name="Honest Mean", line=dict(color='green')))
            fig_mc.add_trace(go.Scatter(x=steps, y=h_mean+2*h_std, fill=None, mode='lines', line_color='rgba(0,255,0,0)', showlegend=False))
            fig_mc.add_trace(go.Scatter(x=steps, y=h_mean-2*h_std, fill='tonexty', mode='lines', line_color='rgba(0,255,0,0.2)', name="Honest 95% CI"))

            # Faulty Mean & CI
            f_mean = np.mean(all_faulty_means, axis=0)
            f_std = np.std(all_faulty_means, axis=0)
            fig_mc.add_trace(go.Scatter(x=steps, y=f_mean, name="Faulty Mean", line=dict(color='red')))
            fig_mc.add_trace(go.Scatter(x=steps, y=f_mean+2*f_std, fill=None, mode='lines', line_color='rgba(255,0,0,0)', showlegend=False))
            fig_mc.add_trace(go.Scatter(x=steps, y=f_mean-2*f_std, fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name="Faulty 95% CI"))

            fig_mc.update_layout(title="MC Aggregated Reputation", xaxis_title="Step", yaxis_title="Reputation")
            st.plotly_chart(fig_mc, use_container_width=True)
    else:
        st.info("Please run `python mc_demo.py` to generate Monte Carlo results.")

# --- Tab 3: DAG Viewer ---
with tab3:
    if sim_data:
        st.header("DAG Topology")
        st.markdown("Visualising the structure of the Distributed Ledger.")

        ledger = sim_data["dag_ledger"]

        # Limit visualisation to prevent browser crash
        max_nodes = st.slider("Max Transactions to Show", 5, 50, 20)

        dot = Digraph(comment='ACCORD DAG')
        dot.attr(rankdir='LR')

        # Simplified DAG parsing
        # Genesis nodes usually have no parents or special IDs
        count = 0
        for tx_hash, tx_list in ledger.items():
            for tx in tx_list:
                if count >= max_nodes: break

                label = f"TX: {tx_hash[:6]}\nScore: {getattr(tx.metadata, 'consensus_score', 'N/A')}"
                color = "green" if getattr(tx.metadata, 'is_confirmed', False) else "red"
                dot.node(tx_hash, label, color=color)

                # Add edges to parents
                if hasattr(tx, 'parents'):
                    for p_hash in tx.parents:
                        if p_hash in ledger:
                            dot.edge(p_hash, tx_hash)

                count += 1
            if count >= max_nodes: break

        st.graphviz_chart(dot)
    else:
        st.info("Simulation data required to visualise DAG.")

