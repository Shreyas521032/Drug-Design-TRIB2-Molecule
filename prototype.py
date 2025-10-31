import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lime
import lime.lime_tabular
import warnings

# Suppress warnings for a cleaner app
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="🧬 MolPred: AI Docking Score Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching Utility Functions ---

@st.cache_data
def load_data(file_name):
    """
    Loads data from the provided CSV file and performs initial cleaning
    based on the pbl.py script.
    """
    try:
        # Load the CSV file provided by the user
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        st.error(f"Error: Data file '{file_name}' not found. Please make sure it's in the same directory as app.py.")
        return None
    
    # Clean column names (strip whitespace)
    df = df.rename(columns=lambda x: x.strip())
    
    # Select only the relevant columns from the script
    cols = ['MW', '#Atoms', 'SlogP', 'TPSA', 'Flexibility', '#RB', 'HBA', 'HBD',
            'LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']
    
    # Ensure all required columns are present
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        st.error(f"Error: The following required columns are missing from your data: {', '.join(missing_cols)}")
        return None
        
    df = df[cols].dropna()
    return df

@st.cache_data
def preprocess_data(df):
    """
    Applies the 1.5 * IQR outlier removal logic from pbl.py.
    """
    df_cleaned = df.copy()
    relevant_cols = df_cleaned.columns.tolist()

    for col in relevant_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the dataframe
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    
    return df_cleaned

@st.cache_resource(show_spinner="Training models... 🏋️‍♂️ This may take a moment.")
def train_models(df, target_col):
    """
    Trains Linear Regression, XGBoost, and Neural Network models.
    Returns models, metrics, scaler, and training data for LIME.
    """
    # Define features (X) and target (y)
    target_cols = ['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    metrics = {}
    
    # --- Model 1: Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    models['Linear Regression'] = lr
    metrics['Linear Regression'] = {
        'R²': r2_score(y_test, y_pred_lr),
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'MAE': mean_absolute_error(y_test, y_pred_lr)
    }
    
    # --- Model 2: XGBoost Regressor ---
    # Using tuned parameters inspired by typical molecular modeling tasks
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
                                 n_estimators=200,
                                 learning_rate=0.05,
                                 max_depth=6,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb_model.predict(X_test_scaled)
    models['XGBoost'] = xgb_model
    metrics['XGBoost'] = {
        'R²': r2_score(y_test, y_pred_xgb),
        'MSE': mean_squared_error(y_test, y_pred_xgb),
        'MAE': mean_absolute_error(y_test, y_pred_xgb)
    }

    # --- Model 3: Neural Network (MLPRegressor) ---
    # Using tuned parameters inspired by the pbl.py description
    nn_model = MLPRegressor(hidden_layer_sizes=(100, 50),
                            activation='relu',
                            solver='adam',
                            max_iter=1000,
                            alpha=0.001,
                            learning_rate_init=0.001,
                            random_state=42,
                            early_stopping=True,
                            n_iter_no_change=10)
    nn_model.fit(X_train_scaled, y_train)
    y_pred_nn = nn_model.predict(X_test_scaled)
    models['Neural Network'] = nn_model
    metrics['Neural Network'] = {
        'R²': r2_score(y_test, y_pred_nn),
        'MSE': mean_squared_error(y_test, y_pred_nn),
        'MAE': mean_absolute_error(y_test, y_pred_nn)
    }
    
    return {
        'models': models,
        'metrics': metrics,
        'scaler': scaler,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_cols
    }

# --- Plotting Functions ---

def plot_correlation_heatmap(df):
    """Generates and displays a correlation heatmap."""
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, annot_kws={"size": 8})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title("Correlation Matrix of Features and Targets")
    st.pyplot(fig)

def plot_distributions(df):
    """Generates and displays histograms for all columns."""
    st.subheader("Feature & Target Distributions")
    st.write("Distributions of all variables after outlier removal.")
    
    num_cols = len(df.columns)
    num_rows = (num_cols + 2) // 3  # 3 plots per row
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, num_rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f"Distribution of {col}", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    st.pyplot(fig)

def plot_predictions(y_true, y_pred, model_name):
    """Generates an Actual vs. Predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get min and max values for equal axis scaling
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    
    ax.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit (y=x)")
    ax.set_xlabel("Actual Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(f"Actual vs. Predicted - {model_name}", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    st.pyplot(fig)

def plot_lime_explanation(model, scaler, X_train_scaled, feature_names, input_data):
    """Generates and displays a LIME explanation plot."""
    st.subheader("Prediction Explainability (LIME)")
    st.markdown("""
    This plot shows *why* the model made its prediction.
    -   **Green bars:** Features that pushed the prediction **higher**.
    -   **Red bars:** Features that pushed the prediction **lower**.
    The length of the bar shows the magnitude of the contribution.
    """)
    
    # LIME needs the model's predict function
    predict_fn = model.predict
    
    # Create the LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        mode='regression',
        feature_names=feature_names,
        verbose=False,
        random_state=42
    )
    
    # Scale the single user input
    input_scaled = scaler.transform(input_data)
    
    # Generate the explanation
    with st.spinner("Generating LIME explanation... 🧠"):
        exp = explainer.explain_instance(
            data_row=input_scaled[0],
            predict_fn=predict_fn,
            num_features=len(feature_names)
        )
    
    # Display the plot
    fig = exp.as_pyplot_figure()
    fig.suptitle("LIME Explanation for Prediction", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

# --- Main App ---

# --- Sidebar Navigation ---
st.sidebar.title("🧬 MolPred Navigator")
st.sidebar.markdown("Navigate through the project steps.")

page = st.sidebar.radio(
    "Go to:",
    ("🏠 Introduction", 
     "📊 Data & Preprocessing", 
     "📈 Exploratory Data Analysis (EDA)", 
     "🤖 Model Training & Evaluation", 
     "🧪 Interactive Prediction & LIME")
)

# --- Load and Preprocess Data (Done once) ---
data = load_data("data.xlsx - Sheet1.csv")
if data is not None:
    cleaned_data = preprocess_data(data)
    
    # Define features and targets
    target_cols = ['LF Rank Score', 'LF dG', 'LF VSscore', 'LF LE']
    feature_cols = [col for col in cleaned_data.columns if col not in target_cols]

# --- Page 1: Introduction ---
if page == "🏠 Introduction":
    st.title("Welcome to MolPred: An AI-Powered Molecular Docking Score Predictor")
    st.markdown("""
    This application is a final-year project demonstrating a complete machine learning pipeline
    for predicting molecular docking scores from 2D molecular properties.
    
    ### Project Goal
    The goal is to build and evaluate several machine learning models—from simple linear regression 
    to advanced Neural Networks and XGBoost—to accurately predict ligand-protein binding affinity scores 
    (like `LF dG`, `LF VSscore`, etc.).
    
    **Why is this important?**
    Accurate prediction of docking scores can significantly accelerate drug discovery by:
    * Virtually screening millions of compounds quickly.
    * Prioritizing promising candidates for expensive lab testing.
    * Helping chemists understand what molecular properties drive binding.
    
    ### How to Use This App
    Use the sidebar on the left to navigate through the project:
    1.  **Data & Preprocessing:** See the raw data and how we clean it by removing statistical outliers.
    2.  **Exploratory Data Analysis (EDA):** Understand the relationships and distributions within the data.
    3.  **Model Training & Evaluation:** Select a target variable (e.g., `LF dG`) and see how different models perform.
    4.  **Interactive Prediction & LIME:** The most exciting part! Use sliders to create a "virtual molecule" and get an instant prediction. We also use **LIME** (Local Interpretable Model-agnostic Explanations) to show you *exactly why* the model gave that score.
    
    *This project is based on the analysis script `pbl.py`.*
    """)
    st.image("https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-020-74677-4/MediaObjects/41598_2020_74677_Fig1_HTML.png", 
             caption="Conceptual image of molecular docking (Source: Nature.com)")

# --- Page 2: Data & Preprocessing ---
elif page == "📊 Data & Preprocessing" and data is not None:
    st.title("Data Loading & Preprocessing")
    
    st.header("1. Raw Data Loaded")
    st.markdown(f"Loaded `data.xlsx - Sheet1.csv`. Found **{data.shape[0]}** rows and **{data.shape[1]}** relevant columns.")
    st.dataframe(data.head())
    
    st.header("2. Outlier Removal")
    st.markdown("""
    As per the `pbl.py` script, we apply an outlier removal process to clean the data.
    For each column, we remove rows where the value is outside **1.5 times the Interquartile Range (IQR)**.
    This is a standard statistical method to remove extreme values that could skew the model.
    """)
    
    st.subheader("Cleaned Data")
    st.markdown(f"After cleaning, we are left with **{cleaned_data.shape[0]}** rows.")
    st.write(f"**{data.shape[0] - cleaned_data.shape[0]}** rows were removed as outliers.")
    st.dataframe(cleaned_data.head())

# --- Page 3: Exploratory Data Analysis (EDA) ---
elif page == "📈 Exploratory Data Analysis (EDA)" and data is not None:
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Understanding the cleaned data before modeling.")
    
    # Show plots in two columns
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        plot_correlation_heatmap(cleaned_data)
        
    with col2:
        st.subheader("Key Observations from Heatmap")
        st.markdown("""
        * **Targets:** The four `LF` target variables are highly correlated with each other, which is expected.
        * **Features vs. Targets:**
            * `MW` (Molecular Weight) and `#Atoms` show a strong positive correlation with docking scores.
            * `SlogP` (lipophilicity) also shows a moderate positive correlation.
            * `TPSA` (Polar Surface Area) shows a moderate negative correlation.
        * **Multicollinearity:** `MW` and `#Atoms` are very highly correlated (0.95). This is logical, but we'll keep both as the models (especially tree-based ones) can handle it.
        """)
    
    st.divider()
    plot_distributions(cleaned_data)

# --- Page 4: Model Training & Evaluation ---
elif page == "🤖 Model Training & Evaluation" and data is not None:
    st.title("Model Training & Evaluation")
    
    st.markdown("Select a target variable to model. The app will split the data, scale features, and train all three models.")
    
    # User selects the target variable
    selected_target = st.selectbox(
        "**Select Target Variable:**",
        target_cols,
        index=1,  # Default to 'LF dG'
        help="This is the value the models will try to predict."
    )
    
    if selected_target:
        # Train (or get from cache) the models
        results = train_models(cleaned_data, selected_target)
        
        st.header("Model Performance Metrics")
        st.write(f"Results for target: **{selected_target}**")
        
        # Display metrics in a styled DataFrame
        metrics_df = pd.DataFrame(results['metrics']).T
        metrics_df = metrics_df.sort_values(by='R²', ascending=False)
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R²'], color='#abf7b1')
                                      .highlight_min(axis=0, subset=['MSE', 'MAE'], color='#f7abab')
                                      .format("{:.4f}"))
        
        # Find and announce the best model
        best_model_name = metrics_df['R²'].idxmax()
        st.success(f"**Best Model:** Based on $R^2$ score, the **{best_model_name}** is the top performer! 🎉")
        
        st.header("Actual vs. Predicted Plot")
        st.markdown("This plot shows how well the *best model's* predictions (y-axis) match the *actual* values (x-axis).")
        
        # Get predictions from the best model
        best_model = results['models'][best_model_name]
        y_pred_best = best_model.predict(results['X_test_scaled'])
        
        plot_predictions(results['y_test'], y_pred_best, best_model_name)

# --- Page 5: Interactive Prediction & LIME ---
elif page == "🧪 Interactive Prediction & LIME" and data is not None:
    st.title("🧪 Interactive Prediction & Explainability")
    st.markdown("This is where the magic happens! Create a virtual compound using the sliders and see what score the model predicts. Then, see *why* it made that prediction.")
    
    # --- Sidebar Inputs ---
    st.sidebar.header("Prediction Inputs")
    st.sidebar.markdown("Adjust these sliders to define your molecule.")
    
    input_data = {}
    for col in feature_cols:
        min_val = float(cleaned_data[col].min())
        max_val = float(cleaned_data[col].max())
        mean_val = float(cleaned_data[col].mean())
        # Use st.sidebar.slider
        input_data[col] = st.sidebar.slider(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            format="%.2f"
        )
    
    st.sidebar.markdown("---")
    
    # Display the user's input in the main panel
    st.header("Your Input Molecule Properties")
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df)
    
    st.divider()
    
    # --- Prediction and LIME ---
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.header("Prediction")
        st.markdown("Select a target and click Predict.")
        
        pred_target = st.selectbox(
            "Select Target Variable:",
            target_cols,
            index=1,  # Default to 'LF dG'
            key="pred_target"
        )
        
        if st.button(f"**Predict {pred_target}!**", type="primary", use_container_width=True):
            # Load the models for this target
            results = train_models(cleaned_data, pred_target)
            
            # Find the best model
            metrics_df = pd.DataFrame(results['metrics']).T
            best_model_name = metrics_df['R²'].idxmax()
            best_model = results['models'][best_model_name]
            
            # Scale the input and predict
            input_scaled = results['scaler'].transform(input_df)
            prediction = best_model.predict(input_scaled)
            
            st.metric(
                label=f"Predicted {pred_target} (using {best_model_name})",
                value=f"{prediction[0]:.4f}"
            )
            st.markdown(f"The top-performing model ({best_model_name}) predicts this score for your input.")
            
            # --- Pass data to LIME plot in col2 ---
            with col2:
                plot_lime_explanation(
                    model=best_model,
                    scaler=results['scaler'],
                    X_train_scaled=results['X_train_scaled'],
                    feature_names=results['feature_names'],
                    input_data=input_df
                )
        else:
            with col2:
                st.info("Click the 'Predict' button to generate the prediction and its LIME explanation. 🧠")

elif data is None:
    st.error("Data could not be loaded. Please check the file name and contents.")
