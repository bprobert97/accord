import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from graphviz import Digraph
from mc_demo import recalculate_all_kpis, NUM_RUNS, NUM_PROCESSES
from accord_demo import DEFAULT_CONFIG
from src.plotting import plot_ground_tracks_plotly

# --- Configuration ---
SIM_RESULTS_PATH = "sim_data/sim_results.npz"
MC_RESULTS_PATH = "sim_data/mc_results/mc_results.npz"

st.set_page_config(page_title="ACCORD Dashboard", layout="wide")

# --- Global Styling ---
st.markdown("""
    <style>
    /* Global font size increase */
    html, body, [class*="st-"] {
        font-size: 1.15rem;
    }
    
    /* Headers */
    h1 { font-size: 3.5rem !important; }
    h2 { font-size: 2.5rem !important; }
    h3 { font-size: 2rem !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        min-width: 350px;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 1.2rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    /* Labels and inputs */
    .stMarkdown p, .stMarkdown li {
        font-size: 1.2rem !important;
    }
    
    .stNumberInput label, .stSlider label {
        font-size: 1.3rem !important;
    }
    
    .stNumberInput input {
        font-size: 1.3rem !important;
    }
    
    .stButton button {
        font-size: 1.3rem !important;
        padding: 0.5rem 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
tab0, tab1, tab2, tab3 = st.tabs(["📋 Configuration", "📊 Results Explorer", "📈 Sensitivity Analysis", "🕸️ DAG Viewer"])

# --- Tab 0: Configuration ---
with tab0:
    st.header("Simulation Parameters")
    st.markdown("Parameters currently set in `DEFAULT_CONFIG` (as defined in `accord_demo.py`).")

    col1, col2 = st.columns(2)

    config_dict = {
        "N (Satellites)": DEFAULT_CONFIG.N,
        "Steps": DEFAULT_CONFIG.steps,
        "Time Step (dt)": f"{DEFAULT_CONFIG.dt} s",
        "Range Noise (sig_r)": f"{DEFAULT_CONFIG.sig_r} m",
        "Range-Rate Noise (sig_rdot)": f"{DEFAULT_CONFIG.sig_rdot} m/s",
        "Process Noise (Target)": DEFAULT_CONFIG.q_acc_target,
        "Process Noise (Observer)": DEFAULT_CONFIG.q_acc_obs,
        "Initial Random Seed": DEFAULT_CONFIG.seed,
        "Number of Monte Carlo Runs": NUM_RUNS,
        "Number of CPU Cores Used": NUM_PROCESSES
    }

    # Split into two columns
    items = list(config_dict.items())
    midpoint = len(items) // 2 + len(items) % 2

    def display_config_item(label, value):
        st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <p style="font-size: 18px; margin-bottom: 0px; color: gray;">{label}</p>
                <p style="font-size: 32px; font-weight: bold; margin-top: -5px;">{value}</p>
            </div>
        """, unsafe_allow_html=True)

    with col1:
        for key, value in items[:midpoint]:
            display_config_item(key, value)

    with col2:
        for key, value in items[midpoint:]:
            display_config_item(key, value)

# --- Tab 1: Results Explorer ---
with tab1:
    if sim_data:
        st.header("Simulation Analysis")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Satellite Reputation")
            rep_history = sim_data["rep_history"]
            faulty_ids = sim_data["faulty_ids"]

            # Convert to DataFrame for Plotly
            rep_df_list = []
            for sid, history in rep_history.items():
                is_faulty = int(sid) in faulty_ids
                for step, rep in enumerate(history):
                    rep_df_list.append({
                        "Timestep": step,
                        "Reputation [-]": rep,
                        "Satellite": f"Sat {sid}",
                        "Status": "Faulty" if is_faulty else "Honest"
                    })
            rep_df = pd.DataFrame(rep_df_list)

            fig_rep = px.line(rep_df, x="Timestep", y="Reputation [-]", color="Satellite",
                             line_dash="Status", color_discrete_sequence=px.colors.qualitative.Safe)
            fig_rep.add_hline(y=0.5, line_dash="dot", annotation_text="Neutral", line_color="gray")
            st.plotly_chart(fig_rep, width='stretch')

        with col2:
            st.subheader("Satellite Ground Tracks")
            if sim_data and "truth" in sim_data:
                # Calculate number of satellites (each state is 6 elements)
                num_sats = sim_data["truth"].shape[1] // 6
                fig_map = plot_ground_tracks_plotly(sim_data["truth"], num_sats)
                st.plotly_chart(fig_map, width='stretch')
            elif os.path.exists("sim_data/orbit_map.png"):
                st.image("sim_data/orbit_map.png", caption="Satellite Ground Tracks (Static Backup)")
            else:
                st.info("Showing current positions. For full ground tracks, see 'accord_demo.py' outputs.")
                st.warning("Orbit data not found.")
    else:
        st.info("Please run `python accord.py` to generate simulation data.")

# --- Tab 2: Sensitivity Analysis ---
with tab2:
    if mc_data:
        st.header("Monte Carlo Sensitivity Analysis")
        st.markdown("Calculate KPIs based on loaded Monte Carlo simulation data.")

        col1, col2 = st.columns([1, 3])

        with col1:
            threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.4, 0.05)
            fpr_offset = st.slider("FPR Offset (Initial % ignored)", 0.0, 0.5, 0.2, 0.05)

            if st.button("Calculate KPIs"):
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
            st.plotly_chart(fig_mc, width='stretch')
    else:
        st.info("Please run `python mc_demo.py` to generate Monte Carlo results.")

# --- Tab 3: DAG Viewer ---
with tab3:
    if sim_data:
        st.header("DAG Topology")
        st.markdown("Visualising the structure of the Distributed Ledger.")

        ledger = sim_data["dag_ledger"]

        # Flatten ledger into a chronological list of transactions
        all_txs = []
        for tx_hash, tx_list in ledger.items():
            for tx in tx_list:
                all_txs.append((tx.metadata.timestamp, tx_hash, tx))
        # Sort by timestamp
        all_txs.sort(key=lambda x: x[0])

        num_all_txs = len(all_txs)

        # Number boxes and a submit button for transaction selection
        with st.form("dag_range_form"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                start_idx = st.number_input("Start Index", min_value=0, max_value=num_all_txs, value=0, step=1)
            with col2:
                end_idx = st.number_input("End Index", min_value=0, max_value=num_all_txs, value=min(20, num_all_txs), step=1)
            with col3:
                st.write("<br>", unsafe_allow_html=True) # Vertical alignment
                submit_button = st.form_submit_button("Update View")

        if end_idx < start_idx:
            st.warning("End Index should be greater than or equal to Start Index.")
            selected_txs = []
        else:
            selected_txs = all_txs[start_idx:end_idx]

        selected_hashes = {tx[1] for tx in selected_txs}

        dot = Digraph(comment='ACCORD DAG')
        dot.attr(rankdir='LR', size='8,5')
        dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')

        # Create nodes
        for _, tx_hash, tx in selected_txs:
            # Format score for readability
            score = getattr(tx.metadata, 'consensus_score', 0.0)
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"

            # Genesis transactions handling
            is_genesis = "Genesis" in tx_hash
            display_hash = tx_hash[:6] if not is_genesis else "Genesis"

            label = f"TX: {display_hash}\nScore: {score_str}"

            # Color coding: Green for confirmed, Red for rejected, Gray for pending/initial
            if getattr(tx.metadata, 'is_confirmed', False):
                fillcolor = "#e8f5e9" # Light green for confirmed
                color = "#2e7d32"
            elif getattr(tx.metadata, 'is_rejected', False):
                fillcolor = "#ffebee" # Light red for rejected
                color = "#c62828"
            else:
                fillcolor = "#f5f5f5" # Gray for pending/initial
                color = "#757575"

            dot.node(tx_hash, label, color=color, fillcolor=fillcolor)

        # Create edges (point from child to parent, but with dir='back' to swap arrowhead)
        for _, tx_hash, tx in selected_txs:
            parent_hashes = getattr(tx.metadata, 'parent_hashes', [])
            for p_hash in parent_hashes:
                # We show the edge if BOTH nodes are in the current view
                if p_hash in selected_hashes:
                    # dot.edge(child, parent, dir='back') draws the arrow pointing to the child
                    # but respects the chronological layout (genesis on left)
                    dot.edge(p_hash, tx_hash, dir='back')

        st.graphviz_chart(dot, width='stretch')
    else:
        st.info("Simulation data required to visualise DAG.")

