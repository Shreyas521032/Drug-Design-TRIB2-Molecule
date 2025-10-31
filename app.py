import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import load_model
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Molecular Docking Prediction Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: rgba(255, 255, 255, 0.95);
    }
    h1 {
        color: #667eea;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2 {
        color: #764ba2;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1>🧬 Molecular Docking Prediction Platform</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 10px; margin-bottom: 30px;'>
    <h3 style='color: #667eea;'>Pre-trained Models for Drug Discovery</h3>
    <p style='font-size: 18px; color: #555;'>Predicting molecular binding affinity using optimized regression models</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(path="data.csv"):
    """Load dataset once and cache it for performance."""
    return pd.read_csv(path)

@st.cache_resource
def load_models():
    """Load all pre-trained models."""
    models = {}
    errors = []
    model_paths = {
        'Random Forest': 'RandomForest_Tuned.pkl',
        'Decision Tree': 'DecisionTree_Tuned.pkl',
        'Neural Network': ['NeuralNetwork_Tuned.pkl', 'NeuralNetwork_Tuned.h5', 'NeuralNetwork_Tuned.keras'],
        'XGBoost': {
            'LF Rank Score': 'XGBoost_Tuned_LF_Rank_Score.pkl',
            'LF dG': 'XGBoost_Tuned_LF_dG.pkl',
            'LF VSscore': 'XGBoost_Tuned_LF_VSscore.pkl',
            'LF LE': 'XGBoost_Tuned_LF_LE.pkl'
        }
    }
    
    # Load RF, DT models
    for model_name in ['Random Forest', 'Decision Tree']:
        try:
            if os.path.exists(model_paths[model_name]):
                models[model_name] = joblib.load(model_paths[model_name])
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")
    
    # Load Neural Network (try multiple formats)
    nn_loaded = False
    for nn_path in model_paths['Neural Network']:
        if os.path.exists(nn_path):
            try:
                if nn_path.endswith('.pkl'):
                    # Try loading as joblib first (sklearn MLPRegressor or similar)
                    models['Neural Network'] = joblib.load(nn_path)
                    nn_loaded = True
                    break
            except:
                try:
                    # Try loading as Keras model
                    models['Neural Network'] = load_model(nn_path)
                    nn_loaded = True
                    break
                except Exception as e:
                    errors.append(f"Neural Network ({nn_path}): {str(e)}")
    
    # Load XGBoost models
    xgb_models = {}
    for target, path in model_paths['XGBoost'].items():
        try:
            if os.path.exists(path):
                xgb_models[target] = joblib.load(path)
        except Exception as e:
            errors.append(f"XGBoost-{target}: {str(e)}")
    
    if xgb_models:
        models['XGBoost'] = xgb_models
    
    # Show errors if any
    if errors:
        with st.sidebar:
            with st.expander("⚠️ Model Loading Warnings", expanded=False):
                for error in errors:
                    st.warning(error)
    
    return models, len(models) > 0

def preprocess_data(df, remove_outliers=True):
    """Clean and preprocess the data."""
    # Clean column names
    df = df.rename(columns=lambda x: x.strip())
    
    # Select relevant columns
    cols = ['MW', '#Atoms', 'SlogP', 'TPSA', 'Flexibility', '#RB', 'HBA', 'HBD',
            'LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']
    df = df[cols].dropna()
    
    original_shape = df.shape
    
    # Remove outliers
    if remove_outliers:
        relevant_cols = cols
        df_cleaned = df.copy()
        
        for col in relevant_cols:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
            df_cleaned = df_cleaned.drop(outliers.index)
        
        cleaned_shape = df_cleaned.shape
        return df_cleaned, original_shape, cleaned_shape
    
    return df, original_shape, df.shape

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # Load models
    models, models_loaded = load_models()
    
    if models_loaded:
        st.success(f"✅ {len(models)} model(s) loaded successfully!")
        st.markdown("**Available Models:**")
        for model_name in models.keys():
            st.markdown(f"- {model_name}")
    else:
        st.error("❌ No models found! Please ensure model files are in the same directory.")
    
    st.markdown("---")
    st.subheader("🎯 Model Selection")
    available_models = list(models.keys())
    model_choice = st.multiselect(
        "Choose Models for Prediction:",
        available_models,
        default=available_models if available_models else []
    )

    st.markdown("---")
    st.subheader("🔧 Analysis Options")
    show_outliers = st.checkbox("Remove Outliers", value=True)

    st.markdown("---")

# Load data
data = load_data("data.csv")

# Main app logic
if data is not None and models_loaded:
    # Preprocessing
    df_processed, orig_shape, clean_shape = preprocess_data(data, show_outliers)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", "🔍 EDA", "📈 Correlation Analysis", 
        "🎯 Single Prediction", "📉 Batch Predictions", "📊 Model Evaluation"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h2>{orig_shape[0]}</h2>
                <p>Original Samples</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h2>{clean_shape[0]}</h2>
                <p>Cleaned Samples</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h2>{orig_shape[0] - clean_shape[0]}</h2>
                <p>Outliers Removed</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h2>{clean_shape[1]}</h2>
                <p>Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📋 Sample Data")
            st.dataframe(df_processed.head(10), use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistical Summary")
            st.dataframe(df_processed.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔢 Feature Descriptions")
        
        feature_info = pd.DataFrame({
            'Feature': ['MW', '#Atoms', 'SlogP', 'TPSA', 'Flexibility', '#RB', 'HBA', 'HBD'],
            'Description': [
                'Molecular Weight',
                'Number of Atoms',
                'Lipophilicity (Water Solubility)',
                'Total Polar Surface Area',
                'Conformational Flexibility',
                'Number of Rotatable Bonds',
                'Hydrogen Bond Acceptors',
                'Hydrogen Bond Donors'
            ],
            'Role': ['Predictor']*8
        })
        
        target_info = pd.DataFrame({
            'Feature': ['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE'],
            'Description': [
                'Ligand Fit Ranking Score',
                'Binding Free Energy',
                'Virtual Screening Score',
                'Ligand Efficiency'
            ],
            'Role': ['Target']*4
        })
        
        st.dataframe(pd.concat([feature_info, target_info]), use_container_width=True)
    
    # Tab 2: EDA
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        features = ['MW', '#Atoms', 'SlogP', 'TPSA', 'Flexibility', '#RB', 'HBA', 'HBD']
        targets = ['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']
        
        with col1:
            st.subheader("📊 Feature Distributions")
            selected_feature = st.selectbox("Select Feature", features)
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
            
            fig.add_trace(
                go.Histogram(x=df_processed[selected_feature], name=selected_feature, 
                           marker_color='#667eea', nbinsx=30),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(y=df_processed[selected_feature], name=selected_feature,
                      marker_color='#764ba2'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Target Distributions")
            selected_target = st.selectbox("Select Target", targets)
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
            
            fig.add_trace(
                go.Histogram(x=df_processed[selected_target], name=selected_target,
                           marker_color='#667eea', nbinsx=30),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Box(y=df_processed[selected_target], name=selected_target,
                      marker_color='#764ba2'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Correlation Analysis
    with tab3:
        st.header("📈 Correlation Analysis")
        
        corr_matrix = df_processed.corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Feature Correlation Heatmap")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔝 Strongest Positive Correlations")
            corr_pairs = corr_matrix.unstack()
            corr_pairs = corr_pairs[corr_pairs < 1]
            top_positive = corr_pairs.nlargest(10)
            st.dataframe(pd.DataFrame({'Correlation': top_positive}))
        
        with col2:
            st.subheader("🔻 Strongest Negative Correlations")
            top_negative = corr_pairs.nsmallest(10)
            st.dataframe(pd.DataFrame({'Correlation': top_negative}))
    
    # Tab 4: Model Predictions
    with tab4:
        st.header("🎯 Model Predictions on Test Data")
        
        if model_choice:
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Prepare data
                    features_to_scale = df_processed.drop(columns=['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']).columns
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(df_processed[features_to_scale])
                    df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale, index=df_processed.index)
                    
                    X = df_scaled
                    y = df_processed[['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']]
                    
                    results = {}
                    targets = ['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']
                    
                    # Random Forest
                    if "Random Forest" in model_choice and "Random Forest" in models:
                        st.write("📊 Predicting with Random Forest...")
                        try:
                            y_pred_rf = models['Random Forest'].predict(X)
                            
                            # Ensure correct shape
                            if len(y_pred_rf.shape) == 1:
                                y_pred_rf = y_pred_rf.reshape(-1, 1)
                            
                            mse_rf = mean_squared_error(y, y_pred_rf, multioutput='raw_values')
                            r2_rf = r2_score(y, y_pred_rf, multioutput='raw_values')
                            mae_rf = mean_absolute_error(y, y_pred_rf, multioutput='raw_values')
                            
                            results['Random Forest'] = {
                                'MSE': mse_rf, 'R2': r2_rf, 'MAE': mae_rf,
                                'predictions': y_pred_rf,
                                'actual': y.values
                            }
                        except Exception as e:
                            st.error(f"Error with Random Forest: {str(e)}")
                    
                    # Decision Tree
                    if "Decision Tree" in model_choice and "Decision Tree" in models:
                        st.write("📊 Predicting with Decision Tree...")
                        try:
                            y_pred_dt = models['Decision Tree'].predict(X)
                            
                            # Ensure correct shape
                            if len(y_pred_dt.shape) == 1:
                                y_pred_dt = y_pred_dt.reshape(-1, 1)
                            
                            mse_dt = mean_squared_error(y, y_pred_dt, multioutput='raw_values')
                            r2_dt = r2_score(y, y_pred_dt, multioutput='raw_values')
                            mae_dt = mean_absolute_error(y, y_pred_dt, multioutput='raw_values')
                            
                            results['Decision Tree'] = {
                                'MSE': mse_dt, 'R2': r2_dt, 'MAE': mae_dt,
                                'predictions': y_pred_dt,
                                'actual': y.values
                            }
                        except Exception as e:
                            st.error(f"Error with Decision Tree: {str(e)}")
                    
                    # XGBoost
                    if "XGBoost" in model_choice and "XGBoost" in models:
                        st.write("📊 Predicting with XGBoost...")
                        try:
                            xgb_preds = []
                            xgb_mse = []
                            xgb_r2 = []
                            xgb_mae = []
                            
                            for target in targets:
                                if target in models['XGBoost']:
                                    pred = models['XGBoost'][target].predict(X)
                                    xgb_preds.append(pred)
                                    xgb_mse.append(mean_squared_error(y[target], pred))
                                    xgb_r2.append(r2_score(y[target], pred))
                                    xgb_mae.append(mean_absolute_error(y[target], pred))
                            
                            if xgb_preds:
                                results['XGBoost'] = {
                                    'MSE': np.array(xgb_mse), 'R2': np.array(xgb_r2), 'MAE': np.array(xgb_mae),
                                    'predictions': np.array(xgb_preds).T,
                                    'actual': y.values
                                }
                        except Exception as e:
                            st.error(f"Error with XGBoost: {str(e)}")
                    
                    # Neural Network
                    if "Neural Network" in model_choice and "Neural Network" in models:
                        st.write("📊 Predicting with Neural Network...")
                        try:
                            y_pred_nn = models['Neural Network'].predict(X, verbose=0)
                            
                            # Ensure correct shape
                            if len(y_pred_nn.shape) == 1:
                                y_pred_nn = y_pred_nn.reshape(-1, 1)
                            
                            mse_nn = mean_squared_error(y, y_pred_nn, multioutput='raw_values')
                            r2_nn = r2_score(y, y_pred_nn, multioutput='raw_values')
                            mae_nn = mean_absolute_error(y, y_pred_nn, multioutput='raw_values')
                            
                            results['Neural Network'] = {
                                'MSE': mse_nn, 'R2': r2_nn, 'MAE': mae_nn,
                                'predictions': y_pred_nn,
                                'actual': y.values
                            }
                        except Exception as e:
                            st.error(f"Error with Neural Network: {str(e)}")
                    
                    st.session_state['results'] = results
                    st.session_state['y_true'] = y
                    st.session_state['targets'] = targets
                    
                    st.success("✅ Predictions generated successfully!")
                    
                    # Show sample predictions
                    st.subheader("📋 Sample Predictions")
                    for model_name, metrics in results.items():
                        st.markdown(f"**{model_name}:**")
                        try:
                            # Handle different prediction shapes
                            preds = metrics['predictions'][:10]
                            if len(preds.shape) == 1:
                                # Single target prediction
                                pred_df = pd.DataFrame(
                                    preds.reshape(-1, 1),
                                    columns=[targets[0]]
                                )
                            else:
                                # Multiple targets
                                pred_df = pd.DataFrame(
                                    preds,
                                    columns=targets[:preds.shape[1]]
                                )
                            pred_df.index.name = "Sample"
                            st.dataframe(pred_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying predictions for {model_name}: {str(e)}")
                        st.markdown("---")
        else:
            st.info("👈 Please select at least one model from the sidebar!")
    
    # Tab 5: Results Comparison
    with tab5:
        st.header("📉 Model Performance Comparison")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            targets = st.session_state['targets']
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, metrics in results.items():
                n_targets = len(metrics['R2']) if hasattr(metrics['R2'], '__len__') else 1
                for i in range(min(n_targets, len(targets))):
                    comparison_data.append({
                        'Model': model_name,
                        'Target': targets[i] if n_targets > 1 else targets[0],
                        'R² Score': metrics['R2'][i] if n_targets > 1 else metrics['R2'],
                        'MSE': metrics['MSE'][i] if n_targets > 1 else metrics['MSE'],
                        'MAE': metrics['MAE'][i] if n_targets > 1 else metrics['MAE']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 R² Score Comparison")
                fig = px.bar(comparison_df, x='Target', y='R² Score', color='Model',
                           barmode='group', title="R² Score by Target and Model")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 MSE Comparison")
                fig = px.bar(comparison_df, x='Target', y='MSE', color='Model',
                           barmode='group', title="MSE by Target and Model")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📋 Detailed Metrics Table")
            st.dataframe(comparison_df.style.background_gradient(cmap='RdYlGn_r', subset=['MSE', 'MAE'])
                        .background_gradient(cmap='RdYlGn', subset=['R² Score']),
                        use_container_width=True)
            
            # Best model for each target
            st.markdown("---")
            st.subheader("🏆 Best Performing Models")
            
            for target in targets:
                target_data = comparison_df[comparison_df['Target'] == target]
                best_model = target_data.loc[target_data['R² Score'].idxmax()]
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); 
                            padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <h4 style='color: #667eea;'>{target}</h4>
                    <p><strong>Best Model:</strong> {best_model['Model']}</p>
                    <p><strong>R² Score:</strong> {best_model['R² Score']:.4f}</p>
                    <p><strong>MSE:</strong> {best_model['MSE']:.4f}</p>
                    <p><strong>MAE:</strong> {best_model['MAE']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Prediction vs Actual plots
            st.markdown("---")
            st.subheader("📈 Prediction vs Actual Values")
            
            selected_target_idx = st.selectbox("Select Target Variable", 
                                               range(len(targets)), 
                                               format_func=lambda x: targets[x])
            
            fig = go.Figure()
            
            for model_name, metrics in results.items():
                fig.add_trace(go.Scatter(
                    x=metrics['actual'][:, selected_target_idx],
                    y=metrics['predictions'][:, selected_target_idx],
                    mode='markers',
                    name=model_name,
                    marker=dict(size=8, opacity=0.6)
                ))
            
            # Add perfect prediction line
            min_val = min(results[list(results.keys())[0]]['actual'][:, selected_target_idx])
            max_val = max(results[list(results.keys())[0]]['actual'][:, selected_target_idx])
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Predicted vs Actual: {targets[selected_target_idx]}",
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Please generate predictions in the 'Model Predictions' tab first!")

else:
    # Welcome/Error screen
    if not models_loaded:
        st.error("""
        ### ❌ Models Not Found!
        
        Please ensure the following model files are in the same directory as this script:
        - `RandomForest_Tuned.pkl`
        - `DecisionTree_Tuned.pkl`
        - `NeuralNetwork_Tuned.pkl`
        - `XGBoost_Tuned_LF_Rank_Score.pkl`
        - `XGBoost_Tuned_LF_dG.pkl`
        - `XGBoost_Tuned_LF_VSscore.pkl`
        - `XGBoost_Tuned_LF_LE.pkl`
        """)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <img src='https://img.icons8.com/fluency/96/000000/molecule.png' width='150'>
            <h2 style='color: #667eea; margin-top: 30px;'>Welcome to the Molecular Docking Prediction Platform!</h2>
            <p style='font-size: 20px; color: #555; margin-top: 20px;'>
                Using pre-trained models to predict molecular binding affinities.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🧬 Molecular Docking Prediction Platform | Powered by Pre-trained ML Models</p>
    <p>Built with Streamlit, Scikit-learn, XGBoost, TensorFlow & Plotly</p>
</div>
""", unsafe_allow_html=True)
