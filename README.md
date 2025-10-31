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

### Example CSV Structure
```csv
MW,#Atoms,SlogP,TPSA,Flexibility,#RB,HBA,HBD,LF Rank Score,LF dG,LF VSscore,LF LE
450.52,32,3.45,75.63,0.25,5,6,2,-8.5,-9.2,0.85,0.35
380.48,28,2.89,68.20,0.22,4,5,1,-7.8,-8.5,0.78,0.32
...
```

## 📂 Project Structure

```
molecular-docking-prediction/
│
├── app.py                              # Main Streamlit application
├── data.csv                            # Dataset file
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── models/                             # Pre-trained model files
│   ├── RandomForest_Tuned.pkl
│   ├── DecisionTree_Tuned.pkl
│   ├── NeuralNetwork_Tuned.pkl
│   ├── XGBoost_Tuned_LF_Rank_Score.pkl
│   ├── XGBoost_Tuned_LF_dG.pkl
│   ├── XGBoost_Tuned_LF_VSscore.pkl
│   └── XGBoost_Tuned_LF_LE.pkl
│
└── screenshots/                        # Application screenshots (optional)
    ├── dashboard.png
    ├── prediction.png
    └── evaluation.png
```

## 📦 Dependencies

### Core Libraries
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
xgboost>=1.7.0
joblib>=1.3.0
```

### Installation
Create a `requirements.txt` file with the following content:

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
scikit-learn==1.3.0
tensorflow==2.13.0
xgboost==2.0.0
joblib==1.3.2
```

Then install with:
```bash
pip install -r requirements.txt
```

## 🖼️ Screenshots

### Dashboard Overview
The main dashboard provides an overview of your dataset with key statistics and visualizations.

### Single Prediction Interface
Interactive form for entering molecular properties and getting instant predictions.

### Model Performance Comparison
Visual comparison of different models' performance across all target variables.

## 🔧 Configuration

### Outlier Removal
Toggle outlier removal from the sidebar:
- **Enabled**: Removes outliers using IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- **Disabled**: Uses all data points

### Model Selection
Select which models to use for predictions from the sidebar:
- Random Forest
- Decision Tree
- Neural Network
- XGBoost

## 🐛 Troubleshooting

### Common Issues

#### 1. Models Not Loading
**Problem**: "❌ No models found!" error

**Solution**:
- Ensure all model files are in the correct directory
- Check file names match exactly
- Verify file permissions

#### 2. Data File Not Found
**Problem**: "⚠️ Data File Not Found!" warning

**Solution**:
- Place `data.csv` in the same directory as `app.py`
- Check file name spelling
- Verify file format is CSV

#### 3. Prediction Errors
**Problem**: Error when making predictions

**Solution**:
- Ensure input values are within reasonable ranges
- Check that all models are loaded successfully
- Verify feature names match the training data

#### 4. Memory Issues
**Problem**: Application runs slowly or crashes

**Solution**:
- Reduce dataset size
- Use fewer models simultaneously
- Increase system memory
- Use a smaller batch size for predictions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing libraries
- Streamlit for the excellent web framework
- scikit-learn, XGBoost, and TensorFlow teams

## 📞 Contact

For questions or support, please contact:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/molecular-docking-prediction/issues)

## 🔮 Future Enhancements

- [ ] Add more machine learning models
- [ ] Implement ensemble predictions
- [ ] Add model explanation features (SHAP values)
- [ ] Support for additional file formats
- [ ] Real-time model training interface
- [ ] Export predictions to various formats
- [ ] API endpoint for programmatic access
- [ ] Docker containerization

---

**Built with ❤️ for Drug Discovery Research**
