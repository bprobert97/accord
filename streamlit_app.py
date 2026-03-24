"""
Streamlit dashboard for the ACCORD project.
Provides interactive visualisation and analysis for
autonomous cooperative consensus orbit determination.
"""
import os
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
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
    /* Global font size increase with media queries */
    html, body, [class*="st-"] {
        font-size: 1.15rem;
    }

    @media screen and (max-width: 1024px) {
        html, body, [class*="st-"] { font-size: 1rem; }
    }

    @media screen and (max-width: 768px) {
        html, body, [class*="st-"] { font-size: 0.9rem; }
    }

    /* Force columns to stack vertically on smaller screens to prevent squashing */
    @media screen and (max-width: 1200px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
        }
    }

    /* Ensure the main content area uses all available width on smaller screens */
    @media screen and (max-width: 1200px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
    }

    /* Headers */

    h1 { font-size: 3.5rem !important; }
    h2 { font-size: 2.5rem !important; }
    h3 { font-size: 2rem !important; }

    @media screen and (max-width: 1024px) {
        h1 { font-size: 2.8rem !important; }
        h2 { font-size: 2.1rem !important; }
        h3 { font-size: 1.7rem !important; }
    }

    @media screen and (max-width: 768px) {
        h1 { font-size: 2.2rem !important; }
        h2 { font-size: 1.8rem !important; }
        h3 { font-size: 1.5rem !important; }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        min-width: 350px;
    }
    @media screen and (max-width: 1200px) {
        [data-testid="stSidebar"] { min-width: 300px; }
    }
    @media screen and (max-width: 768px) {
        [data-testid="stSidebar"] { min-width: unset; }
    }

    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 1.2rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    @media screen and (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem !important;
        }
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

    /* Custom Config Display Classes */
    .config-label {
        font-size: 1.1rem;
        margin-bottom: 0px;
        color: gray;
    }
    .config-value {
        font-size: 2rem;
        font-weight: bold;
        margin-top: -5px;
    }
    @media screen and (max-width: 768px) {
        .config-label { font-size: 0.9rem; }
        .config-value { font-size: 1.5rem; }
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
def load_sim_results(path: str) -> Dict[str, Any] | None:
    """Load single simulation results from a .npz file."""
    if not os.path.exists(path):
        return None
    try:
        # Use a context manager to ensure the file is closed
        with np.load(path, allow_pickle=True) as data:
            # We use .item() to extract the object from the 0-d array
            # Typed access to avoid pylint E1136
            return {
                "dag_ledger": data["dag_ledger"],
                "rep_history": data["rep_history"],
                "truth": data["truth"],
                "faulty_ids": set(data["faulty_ids"])
            }
    except (IOError, ValueError, KeyError) as e:
        st.error(f"Error loading simulation results: {e}")
        return None

@st.cache_data
def load_mc_results(path: str) -> List[Any] | None:
    """Load Monte Carlo results from a .npz file."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            return list(data['results'])
    except (IOError, ValueError, KeyError) as e:
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
tab0, tab1, tab2, tab3, tab4 = st.tabs(["📋 Configuration", "📊 Results Explorer",
                                  "📈 Sensitivity Analysis", "🕸️ DAG Viewer", "📚 Resources"])

# --- Tab 0: Configuration ---
with tab0:
    st.header("Simulation Parameters")
    st.markdown(
        "Parameters currently set in `DEFAULT_CONFIG` "
        "(as defined in `accord_demo.py`)."
    )

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
    MIDPOINT = len(items) // 2 + len(items) % 2

    def display_config_item(item_label: str, item_value: Any) -> None:
        """Display a single configuration item with custom styling."""
        st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <p class="config-label">{item_label}</p>
                <p class="config-value">{item_value}</p>
            </div>
        """, unsafe_allow_html=True)

    with col1:
        for key, value in items[:MIDPOINT]:
            display_config_item(key, value)

    with col2:
        for key, value in items[MIDPOINT:]:
            display_config_item(key, value)

# --- Tab 1: Results Explorer ---
with tab1:
    if sim_data:
        st.header("Simulation Analysis")

        st.subheader("Satellite Reputation")
        rep_history = sim_data["rep_history"].item()
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

        fig_rep = px.line(
            rep_df, x="Timestep", y="Reputation [-]", color="Satellite",
            line_dash="Status",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_rep.add_hline(
            y=0.5, line_dash="dot", annotation_text="Neutral", line_color="gray"
        )
        st.plotly_chart(fig_rep, width="stretch")

        st.divider()

        st.subheader("Satellite Ground Tracks")
        if sim_data and "truth" in sim_data:
            # Calculate number of satellites (each state is 6 elements)
            num_sats = sim_data["truth"].shape[1] // 6
            fig_map = plot_ground_tracks_plotly(sim_data["truth"], num_sats)
            st.plotly_chart(fig_map, width="stretch")
        elif os.path.exists("sim_data/orbit_map.png"):
            st.image(
                "sim_data/orbit_map.png",
                caption="Satellite Ground Tracks (Static Backup)",
                width="stretch"
            )
        else:
            st.info(
                "Showing current positions. For full ground tracks, "
                "see 'accord_demo.py' outputs."
            )
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
                new_results = recalculate_all_kpis(
                    mc_data, detection_threshold=threshold,
                    fpr_offset_percent=fpr_offset
                )
                valid_kpis: List[Dict[str, Any]] = [
                    k for k in new_results if k is not None
                ]

                if valid_kpis:
                    st.subheader("Performance Summary")

                    # Core Metrics
                    ttds = [
                        float(k.get("avg_ttd", 0)) for k in valid_kpis
                        if k.get("avg_ttd") is not None
                    ]
                    worst_ttds = [
                        float(k.get("worst_ttd", 0)) for k in valid_kpis
                        if k.get("worst_ttd") is not None
                    ]
                    fprs = [float(k.get("fpr", 0)) for k in valid_kpis]
                    recalls = [float(k.get("recall", 0)) for k in valid_kpis]
                    precisions = [float(k.get("precision", 0)) for k in valid_kpis]

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mean Recall", f"{np.mean(recalls):.1f}%")
                    m2.metric("Mean Precision", f"{np.mean(precisions):.1f}%")
                    m3.metric("Mean FPR", f"{np.mean(fprs):.1f}%")

                    m4, m5 = st.columns(2)
                    m4.metric("Mean TTD", f"{np.mean(ttds):.1f} steps" if ttds else "N/A")
                    m5.metric(
                        "Worst-Case TTD",
                        f"{np.max(worst_ttds):.1f} steps" if worst_ttds else "N/A"
                    )

                    st.divider()
                    st.subheader("System Robustness & Stability")

                    margins = [float(k.get("detection_margin", 0)) for k in valid_kpis]
                    spreads = [float(k.get("honest_spread", 0)) for k in valid_kpis]
                    flips = [float(k.get("flips", 0)) for k in valid_kpis]
                    h_reps = [float(k.get("final_honest_rep", 0)) for k in valid_kpis]
                    f_reps = [float(k.get("final_faulty_rep", 0)) for k in valid_kpis]

                    c1, c2 = st.columns(2)
                    c1.metric("Detection Margin", f"{np.mean(margins):.3f}")
                    c2.metric("Honest Spread (σ)", f"{np.mean(spreads):.3f}")

                    c3, c4, c5 = st.columns(3)
                    c3.metric("Avg Final Honest Rep", f"{np.mean(h_reps):.3f}")
                    c4.metric("Avg Final Faulty Rep", f"{np.mean(f_reps):.3f}")
                    c5.metric("Avg Flips (Stability)", f"{np.mean(flips):.1f}")

        with col2:
            # We recalculate to ensure valid_kpis_plot is available for the plots
            new_results = recalculate_all_kpis(mc_data, detection_threshold=threshold, \
                                               fpr_offset_percent=fpr_offset)
            valid_kpis_plot: List[Dict[str, Any]] = [k for k in new_results if k is not None]

            if valid_kpis_plot:
                # Plot 1: Reputation Trends
                all_honest_means = np.array([np.mean(k["honest_matrix"], axis=0)
                                           for k in valid_kpis_plot])
                all_faulty_means = np.array([np.mean(k["faulty_matrix"], axis=0)
                                           for k in valid_kpis_plot])
                steps = np.arange(all_honest_means.shape[1])

                fig_mc = go.Figure()
                # Honest Mean & CI
                h_mean = np.mean(all_honest_means, axis=0)
                h_std = np.std(all_honest_means, axis=0)
                fig_mc.add_trace(go.Scatter(
                    x=steps, y=h_mean, name="Honest Mean",
                    line={"color": "green", "width": 3}
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps, y=h_mean + h_std, fill=None, mode="lines",
                    line_color="rgba(0,255,0,0)", showlegend=False
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps, y=h_mean - h_std, fill="tonexty", mode="lines",
                    line_color="rgba(0,255,0,0.1)", name="Honest Pop. Spread (1σ)"
                ))

                # Faulty Mean & CI
                f_mean = np.mean(all_faulty_means, axis=0)
                f_std = np.std(all_faulty_means, axis=0)
                fig_mc.add_trace(go.Scatter(
                    x=steps, y=f_mean, name="Faulty Mean",
                    line={"color": "red", "width": 3}
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps, y=f_mean + f_std, fill=None, mode="lines",
                    line_color="rgba(255,0,0,0)", showlegend=False
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps, y=f_mean - f_std, fill="tonexty", mode="lines",
                    line_color="rgba(255,0,0,0.1)", name="Faulty Pop. Spread (1σ)"
                ))

                fig_mc.add_hline(
                    y=threshold, line_dash="dash", line_color="orange",
                    annotation_text=f"Threshold ({threshold})"
                )
                fig_mc.update_layout(
                    title="Monte Carlo Reputation Trends",
                    xaxis_title="Step", yaxis_title="Reputation",
                    legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99}
                )
                st.plotly_chart(fig_mc, width="stretch")

                # Plot 2: Distributions in Rows
                # Reliability Scatter
                recalls = [k.get("recall", 0) for k in valid_kpis_plot]
                precisions = [k.get("precision", 0) for k in valid_kpis_plot]

                fig_rel = px.scatter(x=recalls, y=precisions, labels={'x': 'Recall (%)',
                                                                        'y': 'Precision (%)'},
                                    title="Reliability (Recall vs Precision)",
                                    range_x=[-5, 105], range_y=[-5, 105])
                fig_rel.add_vline(x=np.mean(recalls), line_dash="dot",
                                    line_color="purple", opacity=0.5)
                fig_rel.add_hline(y=np.mean(precisions), line_dash="dot",
                                    line_color="purple", opacity=0.5)
                st.plotly_chart(fig_rel, width="stretch")

                st.divider()

                # TTD Histogram
                ttds_flat = [k.get("avg_ttd") for k in valid_kpis_plot \
                                if k.get("avg_ttd") is not None]
                if ttds_flat:
                    fig_ttd = px.histogram(x=ttds_flat, nbins=15, labels={'x': 'Steps'},
                                            title="Time to Detection Distribution")
                    st.plotly_chart(fig_ttd, width="stretch")
                else:
                    st.info("No detections occurred.")

                st.divider()

                # FPR Histogram
                fprs_flat = [k.get("fpr", 0) for k in valid_kpis_plot]
                fig_fpr = px.histogram(
                    x=fprs_flat, nbins=15, labels={'x': 'FPR (%)'},
                    title="False Positive Rate Distribution",
                    color_discrete_sequence=['salmon']
                )
                st.plotly_chart(fig_fpr, width="stretch")

    else:
        st.info("Please run `python mc_demo.py` to generate Monte Carlo results.")


# --- Tab 3: DAG Viewer ---
with tab3:
    if sim_data:
        st.header("DAG Topology")
        st.markdown("Visualising the structure of the Distributed Ledger.")

        ledger = sim_data["dag_ledger"].item()

        # Flatten ledger into a chronological list of transactions
        all_txs = []
        for tx_hash, tx_list in ledger.items():
            for tx in tx_list:
                all_txs.append((tx.metadata.timestamp, tx_hash, tx))
        # Sort by timestamp
        all_txs.sort(key=lambda x: x[0])

        NUM_ALL_TXS = len(all_txs)

        # Number boxes and a submit button for transaction selection
        with st.form("dag_range_form"):
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                start_idx = st.number_input(
                    "Start Index", min_value=0, max_value=NUM_ALL_TXS, value=0, step=1
                )
            with col2:
                end_idx = st.number_input(
                    "End Index", min_value=0, max_value=NUM_ALL_TXS,
                    value=min(20, NUM_ALL_TXS), step=1
                )
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

            TX_LABEL = f"TX: {display_hash}\nScore: {score_str}"

            # Color coding: Green for confirmed, Red for rejected, Gray for pending/initial
            if getattr(tx.metadata, 'is_confirmed', False):
                NODE_FILLCOLOUR = "#e8f5e9" # Light green for confirmed
                NODE_COLOUR = "#2e7d32"
            elif getattr(tx.metadata, 'is_rejected', False):
                NODE_FILLCOLOUR = "#ffebee" # Light red for rejected
                NODE_COLOUR = "#c62828"
            else:
                NODE_FILLCOLOUR = "#f5f5f5" # Gray for pending/initial
                NODE_COLOUR = "#757575"

            dot.node(tx_hash, TX_LABEL, color=NODE_COLOUR, fillcolor=NODE_FILLCOLOUR)

        # Create edges (point from child to parent, but with dir='back' to swap arrowhead)
        for _, tx_hash, tx in selected_txs:
            parent_hashes = getattr(tx.metadata, 'parent_hashes', [])
            for p_hash in parent_hashes:
                # We show the edge if BOTH nodes are in the current view
                if p_hash in selected_hashes:
                    # dot.edge(child, parent, dir='back') draws the arrow pointing to the child
                    # but respects the chronological layout (genesis on left)
                    dot.edge(p_hash, tx_hash, dir='back')

        st.graphviz_chart(dot, width="stretch")
    else:
        st.info("Simulation data required to visualise DAG.")

# --- Tab 4: Resources ---
with tab4:
    st.header("Project Resources")

    st.subheader("Source Code")
    st.markdown("""
    The complete source code for the ACCORD project is available on GitHub.

    [![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/bprobert97/accord)

    **License:** GNU General Public License v3.0
    """)

    st.divider()

    st.subheader("Cite this Work")
    st.markdown("""
    If you use this work in your research, please cite it as:

    > B. Probert, bprobert97/accord: v3.0. (Mar. 24, 2026). Python. University of Strathclyde, Glasgow. [DOI: 10.5281/zenodo.19206200](https://doi.org/10.5281/zenodo.19206200)
    """)

    st.divider()

    st.subheader("Related Publications")
    st.markdown("""
    * B. Probert, R. A. Clark, E. Blasch, and M. Macdonald,
      “Cooperative Orbit Determination for Trusted, Autonomous, and Decentralised Satellite Operations,”
      in AIAA SCITECH 2026 Forum, in AIAA SciTech Forum. Orlando,
      Florida: American Institute of Aeronautics and Astronautics, Jan. 2026.
      [doi: 10.2514/6.2026-0825](https://arc.aiaa.org/doi/10.2514/6.2026-0825)
    * B. Probert, R. A. Clark, E. Blasch, and M. Macdonald,
      “A Review of Distributed Ledger Technologies for Satellite Operations,”
      IEEE Access, vol. 13, pp. 123230–123258, 2025,
      [doi: 10.1109/ACCESS.2025.3588688](https://ieeexplore.ieee.org/document/11079570)
    """)
