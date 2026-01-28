"""
Air Quality Forecasting & AQI Classification Dashboard
A comprehensive dashboard for visualizing PM2.5 forecasting, AQI classification,
and semi-supervised learning results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_metrics(filepath):
    """Load metrics from JSON file"""
    if Path(filepath).exists():
        with open(filepath) as f:
            return json.load(f)
    return None

@st.cache_data
def load_predictions(filepath):
    """Load predictions from CSV file"""
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    return None

@st.cache_data
def load_alerts(filepath):
    """Load alerts from CSV file"""
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    return None

# Define paths
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

# Load all data
clf_metrics = load_metrics(DATA_DIR / "metrics.json")
reg_metrics = load_metrics(DATA_DIR / "regression_metrics.json")
arima_summary = load_metrics(DATA_DIR / "arima_pm25_summary.json")
self_train_metrics = load_metrics(DATA_DIR / "metrics_self_training.json")
co_train_metrics = load_metrics(DATA_DIR / "metrics_co_training.json")
sweep_results = load_metrics(RESULTS_DIR / "self_training_sweep_results.json")

# Load CSV data
predictions_sample = load_predictions(DATA_DIR / "predictions_sample.csv")
predictions_self_train = load_predictions(DATA_DIR / "predictions_self_training_sample.csv")
predictions_co_train = load_predictions(DATA_DIR / "predictions_co_training_sample.csv")
predictions_regression = load_predictions(DATA_DIR / "regression_predictions_sample.csv")
arima_predictions = load_predictions(DATA_DIR / "arima_pm25_predictions.csv")
alerts_self_train = load_predictions(DATA_DIR / "alerts_self_training_sample.csv")
alerts_co_train = load_predictions(DATA_DIR / "alerts_co_training_sample.csv")

# Dashboard title
st.markdown("<h1 class='main-header'>🌍 Air Quality Forecasting & AQI Classification</h1>", unsafe_allow_html=True)
st.markdown("Beijing Multi-Site Air Quality Analysis | Supervised & Semi-Supervised Learning")
st.markdown("---")

# Sidebar navigation
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["📋 Overview", "🎯 Classification Models", "📈 Regression Models", 
     "⏰ ARIMA Forecasting", "🤖 Semi-Supervised Learning", "📊 Model Comparison"]
)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "📋 Overview":
    st.markdown("<h2 class='section-header'>Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Samples", "~412,935", delta="Train & Test data")
    with col2:
        st.metric("🎯 AQI Classes", "6", delta="Classification targets")
    with col3:
        st.metric("⏰ Forecasting Models", "3", delta="Supervised, ARIMA, Semi-supervised")
    
    st.markdown("---")
    st.markdown("### 🎯 Project Objectives")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1️⃣ PM2.5 Regression**
        - Predict PM2.5 levels
        - Use lag features and weather data
        - Evaluate with RMSE, MAE, R²
        """)
    with col2:
        st.markdown("""
        **2️⃣ AQI Classification**
        - Classify into 6 AQI levels
        - Generate air quality alerts
        - 12 monitoring stations
        """)
    with col3:
        st.markdown("""
        **3️⃣ Semi-Supervised Learning**
        - Self-Training approach
        - Co-Training with multiple views
        - Handle unlabeled data
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Key Metrics Summary")
    
    if clf_metrics and reg_metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            acc = clf_metrics.get("accuracy", 0)
            st.metric("🎯 Classification Accuracy", f"{acc:.2%}", delta="Baseline supervised")
        with col2:
            rmse = reg_metrics.get("rmse", 0)
            st.metric("📈 Regression RMSE", f"{rmse:.2f}", delta="PM2.5 prediction")
        with col3:
            r2 = reg_metrics.get("r2", 0)
            st.metric("📊 R² Score", f"{r2:.4f}", delta="Regression model")
        with col4:
            if arima_summary:
                mae = arima_summary.get("mae", 0)
                st.metric("⏰ ARIMA MAE", f"{mae:.2f}", delta="ARIMA forecast")

# ============================================================================
# PAGE: CLASSIFICATION MODELS
# ============================================================================
elif page == "🎯 Classification Models":
    st.markdown("<h2 class='section-header'>AQI Classification Models</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Baseline Supervised", "Detailed Analysis"])
    
    with tab1:
        if clf_metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{clf_metrics['accuracy']:.2%}")
            with col2:
                st.metric("F1 Macro", f"{clf_metrics['f1_macro']:.4f}")
            with col3:
                st.metric("Train Samples", f"{clf_metrics['n_train']:,}")
            with col4:
                st.metric("Test Samples", f"{clf_metrics['n_test']:,}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.markdown("### 📊 Confusion Matrix")
            cm = np.array(clf_metrics['confusion_matrix'])
            labels = clf_metrics['labels']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                title="Supervised Classification - Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=500
            )
            st.plotly_chart(fig, width='stretch')
            
            # Per-class metrics
            st.markdown("### 📈 Per-Class Performance Metrics")
            metrics_data = []
            for label in labels:
                if label in clf_metrics['report']:
                    metrics_data.append({
                        'AQI Class': label,
                        'Precision': clf_metrics['report'][label]['precision'],
                        'Recall': clf_metrics['report'][label]['recall'],
                        'F1-Score': clf_metrics['report'][label]['f1-score'],
                        'Support': int(clf_metrics['report'][label]['support'])
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(metrics_df, x='AQI Class', y=['Precision', 'Recall', 'F1-Score'],
                            barmode='group', title='Per-Class Metrics')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                fig = px.bar(metrics_df, x='AQI Class', y='Support', title='Test Set Distribution')
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
            
            st.dataframe(metrics_df, width='stretch')
    
    with tab2:
        if predictions_sample is not None:
            st.markdown("### 📊 Sample Predictions")
            st.dataframe(predictions_sample.head(20), width='stretch')
            
            st.markdown("### 📥 Download Sample Data")
            csv = predictions_sample.to_csv(index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="predictions_sample.csv",
                mime="text/csv"
            )

# ============================================================================
# PAGE: REGRESSION MODELS
# ============================================================================
elif page == "📈 Regression Models":
    st.markdown("<h2 class='section-header'>PM2.5 Regression Model</h2>", unsafe_allow_html=True)
    
    if reg_metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{reg_metrics['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{reg_metrics['rmse']:.2f}")
        with col3:
            st.metric("MAE", f"{reg_metrics['mae']:.2f}")
        with col4:
            st.metric("SMAPE %", f"{reg_metrics['smape_pct']:.2f}%")
        
        st.markdown("---")
        
        # Training info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Samples", f"{reg_metrics['n_train']:,}")
        with col2:
            st.metric("Test Samples", f"{reg_metrics['n_test']:,}")
        
        st.markdown("---")
        
        # Feature importance message
        st.markdown("### 🔑 Model Features")
        st.info(f"""
        **Input Features:** {len(reg_metrics['feature_cols'])} total
        - **Numeric Features:** {len(reg_metrics['numeric_cols'])}
        - **Categorical Features:** {len(reg_metrics['categorical_cols'])}
        
        **Key Features:** PM2.5 (previous values), PM10, SO2, NO2, CO, O3, Temperature, 
        Pressure, Dew Point, Rain, Wind Speed, and temporal features (hour, day, month, year)
        """)
        
        # Categorical and numeric columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Numeric Columns")
            numeric_cols = reg_metrics['numeric_cols'][:15]  # Show first 15
            st.write(", ".join(numeric_cols))
            if len(reg_metrics['numeric_cols']) > 15:
                st.caption(f"... and {len(reg_metrics['numeric_cols']) - 15} more")
        
        with col2:
            st.markdown("#### Categorical Columns")
            if reg_metrics['categorical_cols']:
                st.write(", ".join(reg_metrics['categorical_cols']))
            else:
                st.write("None")
    
    # Predictions visualization
    if predictions_regression is not None:
        st.markdown("---")
        st.markdown("### 📊 Sample Predictions")
        st.dataframe(predictions_regression.head(20), width='stretch')
        
        # Prediction vs Actual visualization if columns exist
        if 'actual' in predictions_regression.columns and 'predicted' in predictions_regression.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=predictions_regression['actual'][:100], 
                                    name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(y=predictions_regression['predicted'][:100], 
                                    name='Predicted', mode='lines'))
            fig.update_layout(
                title='PM2.5: Actual vs Predicted (First 100 samples)',
                yaxis_title='PM2.5 Level',
                xaxis_title='Sample Index',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: ARIMA FORECASTING
# ============================================================================
elif page == "⏰ ARIMA Forecasting":
    st.markdown("<h2 class='section-header'>ARIMA Time Series Forecasting</h2>", unsafe_allow_html=True)
    
    if arima_summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Station", arima_summary['station'])
        with col2:
            st.metric("Best ARIMA Order", str(tuple(arima_summary['best_order'])))
        with col3:
            st.metric("RMSE", f"{arima_summary['rmse']:.2f}")
        with col4:
            st.metric("MAE", f"{arima_summary['mae']:.2f}")
        
        st.markdown("---")
        st.markdown("### 📊 Series Diagnostics")
        
        diag = arima_summary['diagnostics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{diag['mean']:.2f}")
        with col2:
            st.metric("Std Dev", f"{diag['std']:.2f}")
        with col3:
            st.metric("Min Value", f"{diag['min']:.2f}")
        with col4:
            st.metric("Max Value", f"{diag['max']:.2f}")
        
        st.markdown("---")
        st.markdown("### 🔍 Stationarity Tests")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            adf_p = diag['adf_pvalue']
            status = "✅ Stationary" if adf_p < 0.05 else "❌ Non-stationary"
            st.metric("ADF Test p-value", f"{adf_p:.4f}", delta=status)
        with col2:
            kpss_p = diag['kpss_pvalue']
            status = "✅ Stationary" if kpss_p > 0.05 else "❌ Non-stationary"
            st.metric("KPSS Test p-value", f"{kpss_p:.4f}", delta=status)
        with col3:
            st.metric("Observations", f"{diag['n']:,}")
        
        st.markdown("---")
        st.markdown("### 📈 Autocorrelation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lag-24 Autocorr", f"{diag['autocorr_lag_24']:.4f}")
        with col2:
            st.metric("Lag-168 Autocorr", f"{diag['autocorr_lag_168']:.4f}")
        
        st.info("Lag-24 represents 1-day seasonality, Lag-168 represents 1-week seasonality")
    
    # ARIMA Predictions visualization
    if arima_predictions is not None:
        st.markdown("---")
        st.markdown("### 📊 ARIMA Forecasts")
        st.dataframe(arima_predictions.head(20), width='stretch')
        
        if 'actual' in arima_predictions.columns and 'forecast' in arima_predictions.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=arima_predictions['actual'][:200], 
                                    name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(y=arima_predictions['forecast'][:200], 
                                    name='Forecast', mode='lines', 
                                    line=dict(dash='dash')))
            fig.update_layout(
                title='ARIMA: Actual vs Forecasted PM2.5 (First 200 periods)',
                yaxis_title='PM2.5 Level',
                xaxis_title='Time Period',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: SEMI-SUPERVISED LEARNING
# ============================================================================
elif page == "🤖 Semi-Supervised Learning":
    st.markdown("<h2 class='section-header'>Semi-Supervised Learning Analysis</h2>", unsafe_allow_html=True)
    
    method_choice = st.radio(
        "Select Semi-Supervised Method:",
        ["Self-Training", "Co-Training"],
        horizontal=True
    )
    
    if method_choice == "Self-Training":
        if self_train_metrics:
            st.markdown("### 🔄 Self-Training Results")
            
            # Configuration
            st.markdown("#### Configuration")
            config = self_train_metrics.get('st_cfg', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence Threshold (τ)", config.get('tau', 0.9))
            with col2:
                st.metric("Max Iterations", config.get('max_iter', 10))
            with col3:
                st.metric("Min New Labels/Iter", config.get('min_new_per_iter', 20))
            with col4:
                st.metric("Validation Fraction", config.get('val_frac', 0.2))
            
            st.markdown("---")
            
            # Test metrics
            st.markdown("#### Test Set Performance")
            test_metrics = self_train_metrics.get('test_metrics', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.2%}")
            with col2:
                st.metric("F1 Macro", f"{test_metrics.get('f1_macro', 0):.4f}")
            with col3:
                st.metric("Test Samples", f"{test_metrics.get('n_test', 0):,}")
            with col4:
                st.metric("Train Samples", f"{test_metrics.get('n_train', 0):,}")
            
            st.markdown("---")
            
            # Training history
            st.markdown("#### 📊 Iteration History")
            history = self_train_metrics.get('history', [])
            if history:
                history_df = pd.DataFrame(history)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['iter'], y=history_df['val_accuracy'],
                    name='Validation Accuracy', mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=history_df['iter'], y=history_df['val_f1_macro'],
                    name='Validation F1 Macro', mode='lines+markers'
                ))
                fig.update_layout(
                    title='Self-Training: Metrics Across Iterations',
                    xaxis_title='Iteration',
                    yaxis_title='Score',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Pseudo-labels growth
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=history_df['iter'], y=history_df['new_pseudo'],
                    name='New Pseudo-Labels', marker_color='lightblue'
                ))
                fig.add_trace(go.Scatter(
                    x=history_df['iter'], 
                    y=history_df['unlabeled_pool'],
                    name='Remaining Unlabeled',
                    yaxis='y2',
                    line=dict(color='orange', dash='dash')
                ))
                fig.update_layout(
                    title='Self-Training: Pseudo-labeling Progress',
                    xaxis_title='Iteration',
                    yaxis_title='New Pseudo-Labels',
                    yaxis2=dict(title='Unlabeled Pool', overlaying='y', side='right'),
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(history_df, use_container_width=True)
            
            # Confusion matrix for self-training
            st.markdown("---")
            st.markdown("#### 📊 Test Set Confusion Matrix")
            if 'confusion_matrix' in test_metrics:
                cm = np.array(test_metrics['confusion_matrix'])
                labels = test_metrics['labels']
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=labels, y=labels,
                    colorscale='Blues',
                    text=cm, texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(
                    title="Self-Training - Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Co-Training
        if co_train_metrics:
            st.markdown("### 🤝 Co-Training Results")
            
            # Configuration
            st.markdown("#### Configuration")
            config = co_train_metrics.get('ct_cfg', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence Threshold (τ)", config.get('tau', 0.9))
            with col2:
                st.metric("Max Iterations", config.get('max_iter', 10))
            with col3:
                st.metric("Max New Labels/Iter", config.get('max_new_per_iter', 500))
            with col4:
                st.metric("Min New Labels/Iter", config.get('min_new_per_iter', 20))
            
            st.markdown("---")
            
            # Test metrics
            st.markdown("#### Test Set Performance")
            test_metrics = co_train_metrics.get('test_metrics', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.2%}")
            with col2:
                st.metric("F1 Macro", f"{test_metrics.get('f1_macro', 0):.4f}")
            with col3:
                st.metric("Test Samples", f"{test_metrics.get('n_test', 0):,}")
            with col4:
                st.metric("Train Samples", f"{test_metrics.get('n_train', 0):,}")
            
            st.markdown("---")
            
            # Model views
            st.markdown("#### 👁️ Multiple Views Architecture")
            model_info = co_train_metrics.get('model_info', {})
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**View 1 (Pollutants):**")
                st.write(", ".join(model_info.get('view1_cols', [])))
            with col2:
                st.markdown("**View 2 (Weather):**")
                st.write(", ".join(model_info.get('view2_cols', [])))
            
            st.markdown("---")
            
            # Training history
            st.markdown("#### 📊 Iteration History")
            history = co_train_metrics.get('history', [])
            if history:
                history_df = pd.DataFrame(history)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['iter'], y=history_df['val_accuracy'],
                    name='Validation Accuracy', mode='lines+markers'
                ))
                fig.add_trace(go.Scatter(
                    x=history_df['iter'], y=history_df['val_f1_macro'],
                    name='Validation F1 Macro', mode='lines+markers'
                ))
                fig.update_layout(
                    title='Co-Training: Metrics Across Iterations',
                    xaxis_title='Iteration',
                    yaxis_title='Score',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(history_df, use_container_width=True)
            
            # Confusion matrix
            st.markdown("---")
            st.markdown("#### 📊 Test Set Confusion Matrix")
            if 'confusion_matrix' in test_metrics:
                cm = np.array(test_metrics['confusion_matrix'])
                labels = test_metrics['labels']
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm, x=labels, y=labels,
                    colorscale='Greens',
                    text=cm, texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                fig.update_layout(
                    title="Co-Training - Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Hyperparameter sweep results
    if sweep_results:
        st.markdown("---")
        st.markdown("### 🔍 Self-Training Hyperparameter Sweep")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Baseline (Supervised)")
            baseline = sweep_results['baseline']
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Accuracy", f"{baseline['accuracy']:.2%}")
            with col_b:
                st.metric("F1 Macro", f"{baseline['f1_macro']:.4f}")
        
        with col2:
            st.markdown("#### Best Self-Training (τ = {})".format(sweep_results['best_tau']))
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Accuracy", f"{sweep_results['best_tau_accuracy']:.2%}")
            with col_b:
                st.metric("F1 Macro", f"{sweep_results['best_tau_f1_macro']:.4f}")
        
        # Sweep results visualization
        sweep_data = []
        for tau, metrics in sweep_results['self_training_sweep'].items():
            tau_val = float(tau.split('_')[1])
            sweep_data.append({
                'τ (Tau)': tau_val,
                'Accuracy': metrics['accuracy'],
                'F1 Macro': metrics['f1_macro'],
                'Pseudo-Labels': metrics['total_pseudo_labels']
            })
        
        sweep_df = pd.DataFrame(sweep_data).sort_values('τ (Tau)')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_df['τ (Tau)'], y=sweep_df['Accuracy'],
            name='Accuracy', mode='lines+markers', marker_size=10
        ))
        fig.add_trace(go.Scatter(
            x=sweep_df['τ (Tau)'], y=sweep_df['F1 Macro'],
            name='F1 Macro', mode='lines+markers', marker_size=10
        ))
        fig.update_layout(
            title='Self-Training: Performance vs Confidence Threshold (τ)',
            xaxis_title='Confidence Threshold (τ)',
            yaxis_title='Score',
            height=400
        )
        st.plotly_chart(fig, width='stretch')
        
        st.dataframe(sweep_df, width='stretch')

# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================
elif page == "📊 Model Comparison":
    st.markdown("<h2 class='section-header'>Comprehensive Model Comparison</h2>", unsafe_allow_html=True)
    
    st.markdown("### 🔍 Classification Models Comparison")
    
    comparison_data = []
    
    if clf_metrics:
        comparison_data.append({
            'Model': 'Supervised (Baseline)',
            'Accuracy': clf_metrics['accuracy'],
            'F1 Macro': clf_metrics['f1_macro'],
            'Type': 'Supervised'
        })
    
    if self_train_metrics:
        test_m = self_train_metrics.get('test_metrics', {})
        comparison_data.append({
            'Model': 'Self-Training (τ=0.9)',
            'Accuracy': test_m.get('accuracy', 0),
            'F1 Macro': test_m.get('f1_macro', 0),
            'Type': 'Semi-Supervised'
        })
    
    if co_train_metrics:
        test_m = co_train_metrics.get('test_metrics', {})
        comparison_data.append({
            'Model': 'Co-Training (τ=0.9)',
            'Accuracy': test_m.get('accuracy', 0),
            'F1 Macro': test_m.get('f1_macro', 0),
            'Type': 'Semi-Supervised'
        })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(comp_df, x='Model', y='Accuracy', color='Type',
                        title='Classification Accuracy Comparison',
                        color_discrete_map={'Supervised': '#1f77b4', 'Semi-Supervised': '#ff7f0e'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.bar(comp_df, x='Model', y='F1 Macro', color='Type',
                        title='F1 Macro Comparison',
                        color_discrete_map={'Supervised': '#1f77b4', 'Semi-Supervised': '#ff7f0e'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        st.dataframe(comp_df, width='stretch')
    
    st.markdown("---")
    st.markdown("### 📈 Regression Models Comparison")
    
    if reg_metrics:
        st.markdown("#### PM2.5 Regression Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{reg_metrics['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{reg_metrics['rmse']:.2f}")
        with col3:
            st.metric("MAE", f"{reg_metrics['mae']:.2f}")
        with col4:
            st.metric("SMAPE", f"{reg_metrics['smape_pct']:.2f}%")
    
    if arima_summary:
        st.markdown("#### ARIMA Time Series Model")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Order", str(tuple(arima_summary['best_order'])))
        with col2:
            st.metric("RMSE", f"{arima_summary['rmse']:.2f}")
        with col3:
            st.metric("MAE", f"{arima_summary['mae']:.2f}")
        with col4:
            st.metric("AIC", f"{arima_summary['best_score']:.2f}")
    
    st.markdown("---")
    st.markdown("### 🎯 Summary & Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### ✅ Strengths
        - **Supervised model** performs well on classification
        - **Regression R²** indicates good predictive power
        - **Multiple approaches** provide robustness
        """)
    
    with col2:
        st.markdown("""
        #### ⚠️ Observations
        - Semi-supervised models may need tuning
        - Class imbalance affects some classes
        - Temporal patterns captured by ARIMA
        """)
    
    with col3:
        st.markdown("""
        #### 💡 Recommendations
        - Tune τ threshold for semi-supervised
        - Address class imbalance
        - Ensemble model combination
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🌍 Air Quality Forecasting & AQI Classification Dashboard</p>
    <p>Beijing Multi-Site Air Quality | Supervised + Semi-Supervised Learning</p>
    <p><small>Data Source: UCI ML Repository | Dashboard: Streamlit</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
