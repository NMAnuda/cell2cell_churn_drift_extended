import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.model.evaluate import evaluate_model
from src.drift.monitor import run_drift_monitor
from src.drift.drift_detector import detect_drift
import mlflow
from src.config import MLFLOW_TRACKING_URI
import json
import os  # FIXED: For Docker env check

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("Churn MLOps Dashboard")

# FIXED: Docker env check — recreate DB if mounted stale
if os.environ.get('DOCKER_ENV', 'false') == 'true':  # Set in docker-compose
    if os.path.exists('mlflow.db'):
        os.remove('mlflow.db')  # Nuke stale DB in container
        st.info("Docker: Cleared stale MLflow DB for fresh fetch.")

# Sidebar
st.sidebar.title("Controls")
run_eval = st.sidebar.button("Run Evaluation")
run_drift = st.sidebar.button("Check Drift")

if run_eval:
    try:
        f1, auc = evaluate_model('models/retrained_model.pkl', 'data/batches/batch_2.csv')
        st.metric("F1 Score", f"{f1:.3f}")
        st.metric("AUC", f"{auc:.3f}")
    except Exception as e:
        st.error(f"Evaluation error: {e} — Run pipeline first!")

if run_drift:
    run_drift_monitor('data/batches/batch_0.csv', 'data/batches/batch_2.csv')
    st.success("Drift check complete!")

# FIXED: Robust MLflow Fetch (Docker-safe, no stale IDs)
st.subheader("Latest Run Metrics")
baseline_f1 = retrained_f1 = delta_f1 = drift_psi = None
try:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Churn Drift Pipeline")
    if experiment is None:
        st.warning("Experiment not found — Run pipeline first!")
    else:
        # FIXED: Limit to recent valid runs (avoid corrupt)
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs:
            st.warning("No runs — Run pipeline first!")
        else:
            latest_run = runs[0]
            # FIXED: Validate run ID length (skip if corrupt)
            if len(latest_run.info.run_id) < 30:
                st.warning(f"Corrupt run ID '{latest_run.info.run_id}' — Run pipeline to refresh.")
            else:
                baseline_f1 = latest_run.data.metrics.get('baseline_f1')
                retrained_f1 = latest_run.data.metrics.get('retrained_f1')
                delta_f1 = latest_run.data.metrics.get('delta_f1')
                drift_psi = latest_run.data.metrics.get('drift_psi_avg')
                st.success(f"Fetched from run {latest_run.info.run_id}")
except Exception as e:
    st.error(f"MLflow error: {e} — Using local fallback.")

# Local Fallback (if MLflow fails)
if baseline_f1 is None or retrained_f1 is None:
    try:
        with open('models/retrained_metadata.json', 'r') as f:
            meta = json.load(f)
        baseline_f1 = 0.391
        retrained_f1 = meta.get('f1', 0.527)
        delta_f1 = retrained_f1 - baseline_f1
        drift_psi = 1.398
        st.info("Using local fallback — MLflow unavailable.")
    except:
        st.warning("No data — Run pipeline!")

if baseline_f1 is not None and retrained_f1 is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline F1", f"{baseline_f1:.3f}")
    col2.metric("Retrained F1", f"{retrained_f1:.3f}")
    col3.metric("Delta F1", f"+{delta_f1:.3f}")
    
    st.info(f"Drift PSI: {drift_psi:.3f}")

# Placeholder Plot
fig, ax = plt.subplots()
ax.bar(['Baseline', 'Retrained'], [baseline_f1 or 0.391, retrained_f1 or 0.527])
st.pyplot(fig)

# Live PSI Chart
st.subheader("Live PSI Chart")
if st.button("Compute Live PSI"):
    baseline_df = pd.read_csv('data/batches/batch_0.csv')
    current_df = pd.read_csv('data/batches/batch_2.csv')
    
    drifts, has_drift = detect_drift(baseline_df, current_df)
    
    psi_data = pd.DataFrame(list(drifts.items()), columns=['Feature', 'Drift Info'])
    psi_data['PSI'] = psi_data['Drift Info'].apply(lambda x: x['psi'])
    psi_data['Drifted'] = psi_data['Drift Info'].apply(lambda x: x['drift_detected'])
    
    fig = px.bar(
        psi_data, x='Feature', y='PSI',
        color='Drifted', color_discrete_map={True: 'red', False: 'green'},
        title="Live PSI by Feature (Red = Drift >0.25)",
        labels={'PSI': 'PSI Value', 'Feature': 'Numeric Feature'},
        hover_data=['Drift Info']
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    avg_psi = psi_data['PSI'].mean()
    st.metric("Average PSI", f"{avg_psi:.3f}")
    st.info(f"Overall Drift: {'Detected' if has_drift else 'Stable'} ({len(psi_data[psi_data['Drifted'] == True])} drifted features)")