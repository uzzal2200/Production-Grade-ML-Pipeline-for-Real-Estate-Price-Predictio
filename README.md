<div align="center">

### *Production-Grade ML Pipeline for Real Estate Price Prediction*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![ZenML](https://img.shields.io/badge/ZenML-0.64.0-purple?style=flat-square)
![MLFlow](https://img.shields.io/badge/MLFlow-2.15.1-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-green?style=flat-square)
![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)

</div>

### 📑 Quick Navigation

- [📋 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [📚 Usage Guide](#-usage-guide)
- [📁 Project Structure](#-project-structure)
- [🔧 Pipeline Architecture](#-pipeline-architecture)
- [📊 Model Details](#-model-details)
- [📞 Support](#-support--troubleshooting)

---

## 📋 Overview

**Prices Predictor System** is a production-grade machine learning pipeline for predicting house prices using the Ames Housing Dataset. The system follows MLOps best practices with automated data processing, feature engineering, model training, evaluation, and deployment capabilities.

### ✨ Key Features

- 🔄 **End-to-End ML Pipeline** - Data ingestion → Preprocessing → Feature Engineering → Model Training → Evaluation
- 🛠️ **Automated Data Processing** - Missing value handling, outlier detection, feature scaling
- ⚙️ **Advanced Feature Engineering** - Log transformation, feature scaling, categorical encoding
- 📦 **Model Versioning** - ZenML integration for experiment tracking and artifact management
- 📊 **Model Evaluation** - Automatic evaluation metrics (MSE, R² Score)
- 🚀 **Deployment Ready** - MLflow model serving with REST API support
- 💻 **Local Predictions** - Direct model inference without server dependency

### 📊 Model Performance

| Metric | Value |
|:------:|:-------:|
| **R² Score** | 0.9221 ⭐ (92.21% accuracy) |
| **Mean Squared Error** | 0.0109 |
| **Algorithm** | Linear Regression + Preprocessing |
| **Features** | 39 house characteristics |
| **Training Time** | ~7 seconds ⚡ |

---

## 🚀 Quick Start

### 📋 Prerequisites

- Python 3.10+
- Conda or pip
- 2GB free disk space

### 💾 Installation Steps

<table>
<tr>
<td width="50%">

**Step 1️⃣: Create Environment**
```bash
conda create -n ml python=3.10 -y
conda activate ml
```
</td>
<td width="50%">

**Step 2️⃣: Install Dependencies**
```bash
pip install -r requirements.txt
```
</td>
</tr>
<tr>
<td colspan="2">

**Step 3️⃣: Verify Installation**
```bash
python -c "import zenml, mlflow, sklearn; print('✅ Ready to go!')"
```

</td>
</tr>
</table>

---

## 📚 Usage Guide

### 🎯 Train Model

Run the complete ML pipeline with data processing and model training:

```bash
python run_pipeline.py
```

<div align="left">

**Expected Output:**
- ✅ Trained Linear Regression model
- ✅ Evaluation metrics (R², MSE, RMSE)
- ✅ Model artifacts saved to ZenML artifact store
- ⚡ Training time: ~10 seconds
- 📊 Accuracy: 92.21% (R² Score)

</div>

### 🔮 Make Predictions

#### **Option 1: Local Prediction** ⭐ (Recommended)

```bash
python predict_local.py
```

✨ Loads the latest trained model and makes instant predictions without server dependency.

<table>
<tr><td>

**Sample Output:**
```
✓ Model loaded successfully
📊 Processing input features...
▶️ Making prediction...
✓ Predicted Price: $185,234.56
```

</td></tr>
</table>

---

#### **Option 2: Deploy & Serve** (Production Deployment)

```bash
python run_deployment.py
```

🚀 Starts MLflow REST API server at `http://127.0.0.1:8000`

Send prediction requests:
```bash
python sample_predict.py
```

---

#### **Option 3: Stop Deployment**

```bash
python run_deployment.py --stop-service
```

---

---

## 📁 Project Structure

```
prices-predictor-system/
├── README.md                          # Project documentation
├── requirements.txt                    # Python dependencies
├── config.yaml                         # ZenML configuration
│
├── run_pipeline.py                    # Training & evaluation entry point
├── run_deployment.py                  # Deployment & inference pipeline
├── sample_predict.py                  # Sample prediction request
├── predict_local.py                   # Local model prediction
│
├── src/                               # Core ML components
│   ├── ingest_data.py                # Data ingestion (ZIP/CSV support)
│   ├── handle_missing_values.py      # Missing value imputation
│   ├── feature_engineering.py        # Feature transformations
│   ├── outlier_detection.py          # Statistical outlier removal
│   ├── data_splitter.py              # Train-test splitting
│   ├── model_building.py             # Model training pipeline
│   └── model_evaluator.py            # Evaluation metrics
│
├── steps/                             # ZenML pipeline steps
│   ├── data_ingestion_step.py        # Data loading wrapper
│   ├── handle_missing_values_step.py # Missing value step
│   ├── feature_engineering_step.py   # Feature engineering step
│   ├── outlier_detection_step.py     # Outlier removal step
│   ├── data_splitter_step.py         # Train-test split step
│   ├── model_building_step.py        # Model training step
│   ├── model_evaluator_step.py       # Evaluation step
│   ├── predictor.py                  # Inference step
│   ├── dynamic_importer.py           # Batch data loader
│   ├── model_loader.py               # Model loading
│   └── prediction_service_loader.py  # Service connection
│
├── pipelines/                         # ML Pipelines
│   ├── training_pipeline.py          # Training workflow
│   └── deployment_pipeline.py        # Deployment workflow
│
├── analysis/                          # Data analysis notebooks
│   ├── EDA.ipynb                     # Exploratory Data Analysis
│   └── analyze_src/                  # Analysis modules
│       ├── basic_data_inspection.py
│       ├── univariate_analysis.py
│       ├── bivariate_analysis.py
│       ├── multivariate_analysis.py
│       └── missing_values_analysis.py
│
├── data/                              # Dataset directory
│   └── archive/
│       └── AmesHousing.csv           # Housing dataset (1,460 samples)
│
├── extracted_data/                    # Processed data cache
│   └── AmesHousing.csv
│
├── mlruns/                            # MLflow tracking data
│   └── 0/                             # Experiment 0
│       └── [run_ids]/                # Individual run artifacts
│
├── explanations/                      # Design pattern examples
│   ├── factory_design_patter.py
│   ├── strategy_design_pattern.py
│   └── template_design_pattern.py
│
└── tests/                             # Test suite (empty)
```

### Key Directories Explained

| Directory | Purpose |
|-----------|---------|
| `src/` | Reusable ML components and utilities |
| `steps/` | ZenML pipeline steps for orchestration |
| `pipelines/` | High-level ML workflows |
| `data/` | Raw input datasets |
| `extracted_data/` | Processed data cache |
| `mlruns/` | MLflow experiment tracking |
| `analysis/` | Data exploration and analysis |

---

---

## 🔧 Pipeline Architecture

### Training Pipeline Flow

```
Raw Data (CSV)
    ↓
Data Ingestion → Handle Missing Values → Feature Engineering
    ↓                                          ↓
Outlier Detection → Data Splitting
    ↓
Train/Test Split
    ↓
Model Building (Linear Regression)
    ↓
Model Evaluation (R², MSE)
    ↓
Model Registry (ZenML Artifacts)
```

### Deployment Pipeline Flow

```
Training Pipeline (Retrained Model)
    ↓
MLflow Model Deployer
    ↓
REST API Server (http://127.0.0.1:8000)
    ↓
Inference Pipeline
    ↓
Batch Predictions
```

---

---

## 📊 Data Information

### Dataset: Ames Housing

- **Samples**: 1,460 houses
- **Features**: 39 numeric attributes
- **Target**: SalePrice (house price in dollars)
- **Source**: [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### Feature Categories

- **Location Features**: MSSubClass, Neighborhood, etc.
- **Property Features**: LotArea, OverallQual, Condition
- **Structural Features**: YearBuilt, Basement, Stories
- **Facility Features**: Bedrooms, Bathrooms, Fireplaces, Garage
- **Sales Features**: SaleType, SaleCondition, YrSold

---

---

## 🛠️ Configuration

### ZenML Config (`config.yaml`)

```yaml
enable_cache: False                    # Disable caching for reproducibility

settings:
  docker:
    required_integrations:
      - mlflow                         # MLflow integration

model:
  name: prices_predictor               # Model name
  license: Apache 2.0                  # License
  description: Predictor of housing prices
  tags: ["regression", "housing", "price_prediction"]
```

### Requirements

```
click==8.1.3                          # CLI framework
matplotlib==3.7.5                     # Visualization
mlflow==2.15.1                        # Experiment tracking
numpy==1.24.4                         # Numerical computing
pandas==2.0.3                         # Data manipulation
scikit-learn==1.3.2                   # ML algorithms
seaborn==0.13.2                       # Statistical viz
statsmodels==0.14.1                   # Statistical modeling
zenml==0.64.0                         # ML orchestration
```

---

---

## 🔍 Model Details

### Algorithm

**Linear Regression with Preprocessing Pipeline**

```python
Pipeline(
    steps=[
        ('preprocessor', ColumnTransformer(
            [('num', SimpleImputer(strategy='mean'), numerical_cols),
             ('cat', Pipeline([
                 ('imputer', SimpleImputer(strategy='most_frequent')),
                 ('onehot', OneHotEncoder(handle_unknown='ignore'))
             ]), categorical_cols)]
        )),
        ('model', LinearRegression())
    ]
)
```

### Preprocessing Steps

1. **Numerical Features**: Mean imputation
2. **Categorical Features**: Mode imputation + One-Hot Encoding
3. **Outlier Removal**: Statistical outlier detection on target variable
4. **Feature Scaling**: Automatic via preprocessing pipeline

### Evaluation Metrics

- **R² Score** (coefficient of determination): How well predictions fit
- **Mean Squared Error**: Average squared prediction error
- **Cross-validation**: Train-test split (20% test)

---

---

## 📈 Monitoring & Tracking

### MLflow Integration

All experiments are tracked in MLflow:

```bash
# View experiment dashboard
mlflow ui --backend-store-uri ./mlruns
```

Tracked metrics:
- Model parameters (intercept, coefficients)
- Training metrics (R², MSE, RMSE)
- Model artifacts (sklearn pipeline)
- Data signatures

### ZenML Artifact Store

Model artifacts stored at:
```
~/.zenml/local_artifact_store/
    └── [run-id]/
        └── sklearn_pipeline/
            └── model
```

---

---

## 🚨 Troubleshooting

### No Model Found

**Issue**: `❌ No trained model found`

**Solution**:
```bash
python run_pipeline.py  # Train first
python predict_local.py # Then predict
```

### Connection Refused (Deployment)

**Issue**: `ConnectionRefusedError: [WinError 10061]`

**Solution**: Deployment server not running:
```bash
python run_deployment.py  # Start server in background
```

### Missing Dependencies

**Issue**: `ModuleNotFoundError: No module named 'zenml'`

**Solution**:
```bash
pip install -r requirements.txt  # Reinstall all packages
```

---

---

## 📝 Command Reference

| Command | Purpose |
|---------|---------|
| `python run_pipeline.py` | Train & evaluate model |
| `python predict_local.py` | Make local predictions |
| `python run_deployment.py` | Deploy model server |
| `python run_deployment.py --stop-service` | Stop deployment |
| `python sample_predict.py` | Send REST prediction request |
| `mlflow ui` | View experiment tracking dashboard |

---

---

## 🏗️ Design Patterns Used

### Factory Pattern
- **Location**: `src/ingest_data.py`
- **Purpose**: Create appropriate data ingestors (ZIP/CSV) based on file type

### Strategy Pattern
- **Location**: `src/model_evaluator.py`
- **Purpose**: Flexible model evaluation strategies (Regression, Classification)

### Template Pattern
- **Location**: `pipelines/training_pipeline.py`
- **Purpose**: Define ML pipeline structure with flexible steps

---

---

## 📄 License

Apache License 2.0 - See LICENSE file for details

---

## � References & Resources

| Resource | Link |
|:------:|----------|
| ZenML Docs | [Official Documentation](https://docs.zenml.io/) |
| MLflow Docs | [Official Documentation](https://mlflow.org/docs/latest/index.html) |
| scikit-learn | [Official Documentation](https://scikit-learn.org/stable/documentation.html) |
| Dataset | [Ames Housing on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |

---

## 📞 Support & Troubleshooting

**Having Issues?**

| Problem | Solution |
|---------|----------|
| ❌ Module not found | Run: `pip install -r requirements.txt` |
| ❌ No model found | Run: `python run_pipeline.py` first |
| ❌ Connection refused | Start server: `python run_deployment.py` |
| ❌ ZenML errors | Verify config: `zenml stack list` |

**Need Help?**
1. Check the **Troubleshooting** section in this README
2. Review **Project Structure** for file locations
3. Run individual steps to isolate problems
4. Check MLflow dashboard: `mlflow ui`

---

<div align="center">

### 📊 **Project Status**: Production Ready ✅

**Made with ❤️ for ML Excellence**

**Last Updated**: February 10, 2026

[⬆ Back to top](#-prices-predictor-system)

</div>
