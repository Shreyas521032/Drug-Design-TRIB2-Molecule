# 🧬 Molecular Docking Prediction Platform

A comprehensive web-based platform for predicting molecular binding affinity using pre-trained machine learning models. This application enables drug discovery researchers to predict ligand-protein binding scores through an intuitive interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

🔗 **Live Deployed Project:** [https://srs-drug-design-trib2-molecule.streamlit.app](https://srs-drug-design-trib2-molecule.streamlit.app/)

## ✨ Features

### 🎯 Single Molecule Prediction
- **Interactive Input Interface**: Enter molecular properties through user-friendly input fields
- **Smart Defaults**: Pre-filled with dataset mean values for quick testing
- **Range Validation**: Helpful tooltips showing min/max ranges for each feature
- **Multi-Model Support**: Get predictions from multiple models simultaneously
- **Visual Comparison**: Automatic comparison charts across different models

### 📊 Data Analysis & Visualization
- **Data Overview**: 
  - Sample data viewer
  - Statistical summaries
  - Feature descriptions
  - Outlier detection and removal

- **Exploratory Data Analysis (EDA)**:
  - Interactive histograms
  - Box plots for distribution analysis
  - Feature and target variable exploration

- **Correlation Analysis**:
  - Comprehensive correlation heatmap
  - Top positive/negative correlations
  - Interactive visualizations

### 📉 Batch Predictions
- **Batch Processing**: Predict on entire test datasets
- **Performance Metrics**: MSE, R², MAE for each model
- **Sample Predictions**: View first 10 predictions from each model

### 📊 Model Evaluation
- **Performance Comparison**: Side-by-side model comparison charts
- **Best Model Identification**: Automatic identification of best-performing models per target
- **Prediction vs Actual Plots**: Visual validation of model accuracy
- **Residual Analysis**: Detailed residual plots for error analysis

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/molecular-docking-prediction.git
cd molecular-docking-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Model Files
Ensure all pre-trained model files are in the project root directory:
- `RandomForest_Tuned.pkl`
- `DecisionTree_Tuned.pkl`
- `NeuralNetwork_Tuned.pkl` (or `.h5` or `.keras`)
- `XGBoost_Tuned_LF_Rank_Score.pkl`
- `XGBoost_Tuned_LF_dG.pkl`
- `XGBoost_Tuned_LF_VSscore.pkl`
- `XGBoost_Tuned_LF_LE.pkl`

### Step 5: Prepare Dataset
Place your `data.csv` file in the project root directory.

## 🎮 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Making Single Predictions

1. **Select Models**: Choose one or more models from the sidebar
2. **Navigate to Single Prediction Tab**: Click on "🎯 Single Prediction"
3. **Enter Molecular Properties**:
   - Molecular Weight (MW)
   - Number of Atoms (#Atoms)
   - Lipophilicity (SlogP)
   - Total Polar Surface Area (TPSA)
   - Flexibility
   - Number of Rotatable Bonds (#RB)
   - Hydrogen Bond Acceptors (HBA)
   - Hydrogen Bond Donors (HBD)
4. **Click "Predict"**: Get instant predictions for all target variables
5. **View Results**: Compare predictions across different models

### Batch Predictions

1. **Select Models**: Choose models from the sidebar
2. **Navigate to Batch Predictions Tab**: Click on "📉 Batch Predictions"
3. **Click "Generate Batch Predictions"**: Process entire dataset
4. **View Results**: Check performance metrics and sample predictions

### Model Evaluation

1. **Generate Batch Predictions First**: Required before evaluation
2. **Navigate to Model Evaluation Tab**: Click on "📊 Model Evaluation"
3. **Explore Metrics**:
   - R² Score comparisons
   - MSE comparisons
   - Best model identification
   - Prediction vs Actual plots
   - Residual analysis

## 🤖 Model Information

### Supported Models

#### 1. **Random Forest Regressor**
- Multi-output regression
- Ensemble learning approach
- Handles non-linear relationships
- Provides feature importance

#### 2. **Decision Tree Regressor**
- Interpretable predictions
- Handles non-linear relationships
- Fast training and prediction

#### 3. **Neural Network**
- Deep learning approach
- Captures complex patterns
- Supports multiple formats (.pkl, .h5, .keras)

#### 4. **XGBoost Regressor**
- Separate models for each target:
  - LF Rank Score
  - LF dG (Binding Free Energy)
  - LF VSscore (Virtual Screening Score)
  - LF LE (Ligand Efficiency)
- Gradient boosting algorithm
- High performance and accuracy

### Target Variables

| Target | Description |
|--------|-------------|
| **LF Rank Score** | Ligand Fit Ranking Score |
| **LF dG** | Binding Free Energy (kcal/mol) |
| **LF VSscore** | Virtual Screening Score |
| **LF LE** | Ligand Efficiency |

## 📁 Dataset Requirements

### Input Features (8 features)

| Feature | Description | Type |
|---------|-------------|------|
| **MW** | Molecular Weight | Continuous |
| **#Atoms** | Number of Atoms | Integer |
| **SlogP** | Lipophilicity (Water Solubility) | Continuous |
| **TPSA** | Total Polar Surface Area | Continuous |
| **Flexibility** | Conformational Flexibility | Continuous |
| **#RB** | Number of Rotatable Bonds | Integer |
| **HBA** | Hydrogen Bond Acceptors | Integer |
| **HBD** | Hydrogen Bond Donors | Integer |

### Data Format
- **File Format**: CSV
- **File Name**: `data.csv`
- **Required Columns**: All 8 input features + 4 target variables
- **Missing Values**: Automatically removed during preprocessing
- **Outlier Handling**: Optional removal using IQR method

---

**Built with ❤️ for Drug Discovery Research**
