import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import load_model
import joblib
import os
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Molecular Docking Analysis Platform",
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
st.markdown("<h1>🧬 Molecular Docking Analysis Platform</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 10px; margin-bottom: 30px;'>
    <h3 style='color: #667eea;'>Advanced Machine Learning for Drug Discovery</h3>
    <p style='font-size: 18px; color: #555;'>Predicting molecular binding affinity using state-of-the-art regression models</p>
</div>
""", unsafe_allow_html=True)

def load_data(path="data.csv"):
    """Load dataset once and cache it for performance."""
    return pd.read_csv(path)
    
# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/molecule.png", width=80)
    st.title("⚙️ Control Panel")

    # Internally load dataset (no user interaction)
    data = load_data("data.csv")

    st.markdown("---")
    st.subheader("🎯 Model Selection")
    model_choice = st.multiselect(
        "Choose Models to Train:",
        ["Random Forest", "Decision Tree", "XGBoost", "Neural Network"],
        default=["Random Forest", "XGBoost"]
    )

    st.markdown("---")
    st.subheader("🔧 Analysis Options")
    show_outliers = st.checkbox("Remove Outliers", value=True)
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100

    st.markdown("---")    

# Load data function
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

# Preprocessing function
def preprocess_data(df, remove_outliers=True):
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

# Main app logic
if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)
    
    # Preprocessing
    df_processed, orig_shape, clean_shape = preprocess_data(df, show_outliers)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", "🔍 EDA", "📈 Correlation Analysis", 
        "🤖 Model Training", "📉 Results Comparison", "🎯 Predictions"
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
    
    # Tab 4: Model Training
    with tab4:
        st.header("🤖 Model Training & Evaluation")
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take a moment."):
                # Prepare data
                features_to_scale = df_processed.drop(columns=['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']).columns
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(df_processed[features_to_scale])
                df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale, index=df_processed.index)
                
                X = df_scaled
                y = df_processed[['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                results = {}
                
                # Random Forest
                if "Random Forest" in model_choice:
                    progress_bar = st.progress(0)
                    st.write("Training Random Forest...")
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train, y_train)
                    y_pred_rf = rf_model.predict(X_test)
                    
                    mse_rf = mean_squared_error(y_test, y_pred_rf, multioutput='raw_values')
                    r2_rf = r2_score(y_test, y_pred_rf, multioutput='raw_values')
                    mae_rf = mean_absolute_error(y_test, y_pred_rf, multioutput='raw_values')
                    
                    results['Random Forest'] = {
                        'MSE': mse_rf, 'R2': r2_rf, 'MAE': mae_rf,
                        'predictions': y_pred_rf
                    }
                    progress_bar.progress(25)
                
                # Decision Tree
                if "Decision Tree" in model_choice:
                    st.write("Training Decision Tree...")
                    dt_model = DecisionTreeRegressor(random_state=42)
                    dt_model.fit(X_train, y_train)
                    y_pred_dt = dt_model.predict(X_test)
                    
                    mse_dt = mean_squared_error(y_test, y_pred_dt, multioutput='raw_values')
                    r2_dt = r2_score(y_test, y_pred_dt, multioutput='raw_values')
                    mae_dt = mean_absolute_error(y_test, y_pred_dt, multioutput='raw_values')
                    
                    results['Decision Tree'] = {
                        'MSE': mse_dt, 'R2': r2_dt, 'MAE': mae_dt,
                        'predictions': y_pred_dt
                    }
                    progress_bar.progress(50)
                
                # XGBoost
                if "XGBoost" in model_choice:
                    st.write("Training XGBoost...")
                    xgb_preds = []
                    xgb_mse = []
                    xgb_r2 = []
                    xgb_mae = []
                    
                    for target in y_train.columns:
                        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
                        xgb_model.fit(X_train, y_train[target])
                        pred = xgb_model.predict(X_test)
                        xgb_preds.append(pred)
                        xgb_mse.append(mean_squared_error(y_test[target], pred))
                        xgb_r2.append(r2_score(y_test[target], pred))
                        xgb_mae.append(mean_absolute_error(y_test[target], pred))
                    
                    results['XGBoost'] = {
                        'MSE': np.array(xgb_mse), 'R2': np.array(xgb_r2), 'MAE': np.array(xgb_mae),
                        'predictions': np.array(xgb_preds).T
                    }
                    progress_bar.progress(75)
                
                # Neural Network
                if "Neural Network" in model_choice:
                    st.write("Training Neural Network...")
                    nn_model = Sequential([
                        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                        Dense(32, activation='relu'),
                        Dense(y_train.shape[1])
                    ])
                    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                    
                    y_pred_nn = nn_model.predict(X_test)
                    mse_nn = mean_squared_error(y_test, y_pred_nn, multioutput='raw_values')
                    r2_nn = r2_score(y_test, y_pred_nn, multioutput='raw_values')
                    mae_nn = mean_absolute_error(y_test, y_pred_nn, multioutput='raw_values')
                    
                    results['Neural Network'] = {
                        'MSE': mse_nn, 'R2': r2_nn, 'MAE': mae_nn,
                        'predictions': y_pred_nn
                    }
                    progress_bar.progress(100)
                
                st.session_state['results'] = results
                st.session_state['y_test'] = y_test
                st.session_state['targets'] = targets
                
                st.success("✅ All models trained successfully!")
    
    # Tab 5: Results Comparison
    with tab5:
        st.header("📉 Model Performance Comparison")
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            y_test = st.session_state['y_test']
            targets = st.session_state['targets']
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, metrics in results.items():
                for i, target in enumerate(targets):
                    comparison_data.append({
                        'Model': model_name,
                        'Target': target,
                        'R² Score': metrics['R2'][i],
                        'MSE': metrics['MSE'][i],
                        'MAE': metrics['MAE'][i]
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
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👆 Please train models in the 'Model Training' tab first!")
    
    # Tab 6: Predictions
    with tab6:
        st.header("🎯 Make Predictions")
        
        if 'results' in st.session_state:
            st.subheader("🔮 Input Molecular Properties")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mw = st.number_input("Molecular Weight (MW)", 200.0, 600.0, 400.0)
                atoms = st.number_input("Number of Atoms", 15, 40, 25)
            
            with col2:
                slogp = st.number_input("SlogP", 1.0, 7.0, 4.0)
                tpsa = st.number_input("TPSA", 40.0, 150.0, 80.0)
            
            with col3:
                flexibility = st.number_input("Flexibility", 0.5, 10.0, 3.0)
                rb = st.number_input("Rotatable Bonds", 2, 12, 5)
            
            with col4:
                hba = st.number_input("H-Bond Acceptors", 3, 12, 6)
                hbd = st.number_input("H-Bond Donors", 1, 5, 2)
            
            if st.button("🚀 Predict Docking Scores", type="primary"):
                st.success("Prediction feature coming soon! This will predict binding affinity based on your input.")
        else:
            st.info("👆 Please train models first!")

else:
    # Welcome screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <img src='https://img.icons8.com/fluency/96/000000/molecule.png' width='150'>
        <h2 style='color: #667eea; margin-top: 30px;'>Welcome to the Molecular Docking Analysis Platform!</h2>
        <p style='font-size: 20px; color: #555; margin-top: 20px;'>
            Upload your molecular dataset to begin analyzing binding affinities and training prediction models.
        </p>
        <div style='margin-top: 40px; padding: 30px; background: rgba(102, 126, 234, 0.1); border-radius: 10px;'>
            <h3 style='color: #764ba2;'>✨ Key Features:</h3>
            <ul style='text-align: left; font-size: 18px; color: #555; max-width: 600px; margin: 20px auto;'>
                <li>📊 Comprehensive data exploration and visualization</li>
                <li>🧬 Advanced molecular property analysis</li>
                <li>🤖 Multiple machine learning models (RF, DT, XGBoost, NN)</li>
                <li>📈 Interactive correlation analysis</li>
                <li>🎯 Real-time predictions and comparisons</li>
                <li>📉 Detailed performance metrics</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>🧬 Molecular Docking Analysis Platform | Powered by Machine Learning</p>
    <p>Built with Streamlit, Scikit-learn, XGBoost, TensorFlow & Plotly</p>
</div>
""", unsafe_allow_html=True)
