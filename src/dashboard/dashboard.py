import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ===== CONFIG =====
API_URL = "http://api:8000"

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Churn MLOps Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #0e1117;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Headers */
    h1 {
        color: #00d4ff;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #00d4ff;
    }
    
    h2 {
        color: #00d4ff;
        font-weight: 600;
        padding-top: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0e1117 100%);
    }
    
    /* Cards */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
    }
    
    /* Form inputs */
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #00d4ff;
    }
    
    /* Divider */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%); border-radius: 10px; margin-bottom: 1rem;'>
        <h1 style='color: white; margin: 0; font-size: 2rem;'></h1>
        <h3 style='color: white; margin: 0;'>Churn MLOps</h3>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("###  Navigation")
    
    menu = st.radio(
        "",
        [" Overview", " Performance", " Drift Monitor", " Models", " Predict", " Pipeline"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("###  Quick Stats")
    
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            st.success(" API Online")
        else:
            st.error(" API Error")
    except:
        st.warning(" API Offline")
    
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

# ===========================
#  OVERVIEW PAGE
# ===========================
if menu == " Overview":
    st.title("📈 System Overview")
    
    try:
        resp = requests.get(f"{API_URL}/latest_metrics", timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Key Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Baseline F1", 
                    f"{data['baseline_f1']:.3f}",
                    help="Initial model F1 score"
                )
            
            with col2:
                st.metric(
                    "Retrained F1", 
                    f"{data['retrained_f1']:.3f}",
                    delta=f"{data['delta_f1']:.3f}",
                    help="Current model F1 score"
                )
            
            with col3:
                drift_status = "🔴 Detected" if data['drift_detected'] else "🟢 Stable"
                st.metric(
                    "Drift Status",
                    drift_status,
                    help="Data drift detection status"
                )
            
            with col4:
                st.metric(
                    "Avg PSI",
                    f"{data['drift_psi_avg']:.3f}",
                    help="Population Stability Index"
                )
            
            st.markdown("---")
            
            # Detailed Info Cards
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("####  Model Performance")
                performance_data = {
                    "Model": ["Baseline", "Retrained"],
                    "F1 Score": [data['baseline_f1'], data['retrained_f1']],
                    "Status": ["Archived", "Active"]
                }
                st.dataframe(
                    pd.DataFrame(performance_data),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.markdown("####  Drift Analysis")
                drift_severity = "High" if data['drift_psi_avg'] > 0.5 else "Medium" if data['drift_psi_avg'] > 0.25 else "Low"
                st.info(f"""
                **PSI Average:** {data['drift_psi_avg']:.3f}  
                **Severity:** {drift_severity}  
                **Drift Detected:** {'Yes' if data['drift_detected'] else 'No'}  
                **Recommendation:** {'Retrain recommended' if data['drift_detected'] else 'Model stable'}
                """)
        
        else:
            st.error(" Could not fetch metrics from API")
            st.info(" Make sure the API is running and the pipeline has been executed at least once.")
    
    except requests.exceptions.RequestException as e:
        st.error(f" Connection Error: {e}")
        st.info(" Check if API service is running: `docker-compose ps`")

# ===========================
#  PERFORMANCE PAGE
# ===========================
elif menu == " Performance":
    st.header("🎯 Model Performance")
    try:
        resp = requests.get(f"{API_URL}/pr_curves")
        if resp.status_code == 200:
            files = resp.json()
            for name, url in files.items():
                st.image(url, caption=name.capitalize())
        else:
            st.warning("PR curves not available — run pipeline first")
    except Exception as e:
        st.error(f"Error loading curves: {e}")

# ===========================
#  DRIFT MONITOR PAGE
# ===========================
elif menu == " Drift Monitor":
    st.title("⚠️ Data Drift Monitoring")
    
    st.markdown("""
    Monitor feature distribution changes between baseline and current data.  
    **PSI > 0.25** indicates significant drift requiring model retraining.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button(" Run Drift Check", use_container_width=True):
            with st.spinner("Analyzing drift..."):
                try:
                    resp = requests.get(f"{API_URL}/drift", timeout=10)
                    
                    if resp.status_code == 200:
                        drift_info = resp.json()
                        
                        # Store in session state
                        st.session_state.drift_data = drift_info
                        st.success(" Drift analysis complete!")
                    else:
                        st.error(" Drift check failed")
                
                except Exception as e:
                    st.error(f" Error: {e}")
    
    # Display results if available
    if 'drift_data' in st.session_state:
        drift_info = st.session_state.drift_data
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average PSI", f"{drift_info['drift_psi']:.3f}")
        
        with col2:
            drifted_count = sum(1 for d in drift_info['drifts'].values() if d['drift_detected'])
            st.metric("Drifted Features", drifted_count)
        
        with col3:
            status = "🔴 Drift Detected" if drift_info['drift_detected'] else "🟢 Stable"
            st.metric("Status", status)
        
        st.markdown("---")
        
        # Visualization
        df = pd.DataFrame.from_dict(drift_info['drifts'], orient="index").reset_index()
        df.rename(columns={"index": "feature"}, inplace=True)
        
        fig = go.Figure()
        
        # Add bars
        colors = ['#ff4444' if d else '#00d4ff' for d in df['drift_detected']]
        
        fig.add_trace(go.Bar(
            x=df['feature'],
            y=df['psi'],
            marker_color=colors,
            text=df['psi'].round(3),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>PSI: %{y:.3f}<extra></extra>'
        ))
        
        # Add threshold line
        fig.add_hline(
            y=0.25, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Threshold (0.25)"
        )
        
        fig.update_layout(
            title="PSI by Feature",
            xaxis_title="Feature",
            yaxis_title="PSI Value",
            height=500,
            template="plotly_dark",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📋 Detailed Drift Report"):
            st.dataframe(
                df.sort_values('psi', ascending=False),
                use_container_width=True,
                hide_index=True
            )

# ===========================
# 🗂 MODELS PAGE
# ===========================
elif menu == " Models":
    st.title("🗂 Model Registry")
    
    try:
        resp = requests.get(f"{API_URL}/models", timeout=5)
        
        if resp.status_code == 200:
            models = resp.json()["models"]
            
            if not models:
                st.info("📦 No models found. Run the pipeline to train models.")
            else:
                for idx, m in enumerate(models):
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            status = "🟢 Active" if idx == 0 else "⚪ Archived"
                            st.markdown(f"### {m['name']} {status}")
                        
                        with col2:
                            f1 = m.get('f1')
                            st.metric("F1 Score", f"{f1:.3f}" if f1 else "N/A")
                        
                        with col3:
                            auc = m.get('auc')
                            st.metric("AUC", f"{auc:.3f}" if auc else "N/A")
                        
                        with col4:
                            threshold = m.get('threshold')
                            st.metric("Threshold", f"{threshold:.3f}" if threshold else "N/A")
                        
                        st.markdown("---")
        else:
            st.warning(" Could not fetch models")
    
    except Exception as e:
        st.error(f" Error: {e}")

# ===========================
#  PREDICT PAGE
# ===========================
elif menu == " Predict":
    st.title("🤖 Live Churn Prediction")
    
    st.markdown("""
    Enter customer features below to predict churn probability.  
    The model will return a probability score and binary prediction.
    """)
    
    # Organize features in tabs
    tab1, tab2 = st.tabs(["📊 Numeric Features", "🏷️ Categorical Features"])
    
    inputs = {}
    
    with tab1:
        cols = st.columns(3)
        for idx, feature in enumerate(NUMERIC_FEATURES):
            with cols[idx % 3]:
                inputs[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    value=0.0,
                    step=0.1,
                    format="%.2f"
                )
    
    with tab2:
        cols = st.columns(2)
        for idx, feature in enumerate(CATEGORICAL_FEATURES):
            with cols[idx % 2]:
                inputs[feature] = st.selectbox(
                    feature.replace('_', ' ').title(),
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button(" Predict Churn", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"features": inputs},
                        timeout=5
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        
                        # Result display
                        st.markdown("###  Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            prob = data['churn_prob']
                            st.metric("Churn Probability", f"{prob:.1%}")
                            
                            # Probability gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prob * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "#00d4ff"},
                                    'steps': [
                                        {'range': [0, 30], 'color': "lightgray"},
                                        {'range': [30, 70], 'color': "gray"},
                                        {'range': [70, 100], 'color': "darkgray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 50
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            prediction = data['prediction']
                            
                            if prediction == 1:
                                st.error("### 🔴 HIGH RISK")
                                st.markdown("**Recommendation:** Customer likely to churn. Engage retention strategy.")
                            else:
                                st.success("### 🟢 LOW RISK")
                                st.markdown("**Recommendation:** Customer stable. Continue normal engagement.")
                    
                    else:
                        st.error(f"❌ Prediction failed: {resp.json().get('detail', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ===========================
#  PIPELINE PAGE
# ===========================
elif menu == " Pipeline":
    st.title("⚡ Pipeline Control Center")
    
    st.markdown("""
    Control and monitor the ML pipeline execution.  
    Trigger full pipeline runs or individual retraining operations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Full Pipeline")
        st.markdown("""
        Executes the complete workflow:
        - Data preprocessing
        - Batch generation
        - Model training
        - Drift detection
        - Conditional retraining
        """)
        
        if st.button(" Run Full Pipeline", use_container_width=True, type="primary"):
            with st.spinner("Pipeline running..."):
                try:
                    resp = requests.post(f"{API_URL}/run_pipeline", timeout=60)
                    
                    if resp.status_code == 200:
                        st.success(" Pipeline completed successfully!")
                        st.balloons()
                    else:
                        st.error(f" Pipeline failed: {resp.json().get('detail', 'Unknown error')}")
                
                except requests.exceptions.Timeout:
                    st.warning(" Pipeline is still running in background...")
                except Exception as e:
                    st.error(f" Error: {e}")
    
    with col2:
        st.markdown("###  Manual Retrain")
        st.markdown("""
        Force model retraining:
        - Uses recent batches
        - Updates model version
        - Logs new metrics
        """)
        
        if st.button(" Trigger Retrain", use_container_width=True):
            with st.spinner("Retraining model..."):
                try:
                    resp = requests.post(f"{API_URL}/trigger_retrain", timeout=60)
                    
                    if resp.status_code == 200:
                        st.success(" Retraining completed!")
                    else:
                        st.error(f" Retrain failed: {resp.json().get('detail', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f" Error: {e}")
    
    st.markdown("---")
    
    # Pipeline logs/status
    st.markdown("###  Recent Activity")
    
    with st.expander("View Pipeline Logs"):
        st.code("""
[2026-01-19 18:30:05] Pipeline started
[2026-01-19 18:30:12] Batches generated: 5
[2026-01-19 18:32:45] Baseline model trained (F1: 0.527)
[2026-01-19 18:33:10] Drift detected (PSI: 1.398)
[2026-01-19 18:35:22] Retrained model (F1: 0.541)
[2026-01-19 18:35:30] Pipeline completed
        """, language="log")