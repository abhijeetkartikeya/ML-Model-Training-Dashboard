# Simplified Code Structure - Quick Reference

## 📁 New File Structure

```
project2/
├── data/
│   └── sample_churn_data.csv
├── src/
│   ├── data_loader.py      # Load data from CSV
│   ├── preprocessor.py     # Clean and prepare data
│   ├── model_trainer.py    # Train ML models
│   ├── evaluator.py        # Calculate metrics
│   └── app.py              # Streamlit dashboard
├── streamlit_app.py        # Main app (copy of src/app.py)
├── requirements.txt
└── README.md
```

## 🚀 How to Use (For Interview)

### 1. Load Data
```python
from src.data_loader import load_sample_data

# Load the sample dataset
df = load_sample_data()
```

### 2. Prepare Data
```python
from src.preprocessor import prepare_data, preprocess_features

# Split into train/test
X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2)

# Preprocess features
X_train_processed, X_test_processed = preprocess_features(X_train, X_test)
```

### 3. Train Models
```python
from src.model_trainer import train_logistic_regression, train_random_forest, make_predictions

# Train Logistic Regression
lr_model = train_logistic_regression(X_train_processed, y_train)

# Train Random Forest
rf_model = train_random_forest(X_train_processed, y_train)

# Make predictions
y_pred, y_proba = make_predictions(rf_model, X_test_processed)
```

### 4. Evaluate
```python
from src.evaluator import calculate_metrics, plot_confusion_matrix

# Calculate metrics
metrics = calculate_metrics(y_test, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

# Plot confusion matrix
fig = plot_confusion_matrix(metrics['confusion_matrix'])
```

## 🎯 Interview Talking Points

### 1. Data Loading (`data_loader.py`)
**What it does:** Loads customer data from CSV files

**Key function:**
- `load_sample_data()` - Loads the sample dataset

**Interview explanation:**
> "I created a simple data loader that reads customer data from a CSV file. It's straightforward - just uses pandas to load the data and prints basic info about what was loaded."

### 2. Preprocessing (`preprocessor.py`)
**What it does:** Cleans data and prepares it for machine learning

**Key functions:**
- `prepare_data()` - Splits data into training and test sets
- `preprocess_features()` - Encodes categories and scales numbers

**Interview explanation:**
> "The preprocessor does three main things: First, it converts the 'Churn' column to 0s and 1s. Then it splits the data 80/20 for training and testing. Finally, it encodes categorical variables using Label Encoding and scales numeric features using StandardScaler."

### 3. Model Training (`model_trainer.py`)
**What it does:** Trains machine learning models

**Key functions:**
- `train_logistic_regression()` - Trains a Logistic Regression model
- `train_random_forest()` - Trains a Random Forest model
- `make_predictions()` - Makes predictions with a trained model

**Interview explanation:**
> "I keep it simple with two models. Logistic Regression is the baseline - it's fast and easy to interpret. Random Forest is more powerful because it uses multiple decision trees. Both are from scikit-learn, which is the standard ML library in Python."

### 4. Evaluation (`evaluator.py`)
**What it does:** Measures how good the models are

**Key functions:**
- `calculate_metrics()` - Calculates Accuracy and ROC-AUC
- `plot_confusion_matrix()` - Shows predictions vs actual
- `get_feature_importance()` - Shows which features matter most

**Interview explanation:**
> "For evaluation, I focus on two key metrics: Accuracy tells us how often we're right overall, and ROC-AUC measures how well we separate churners from non-churners. The confusion matrix visualization helps us see where the model makes mistakes."

### 5. Dashboard (`app.py`)
**What it does:** Interactive web interface using Streamlit

**Interview explanation:**
> "The dashboard is built with Streamlit, which makes it really easy to create ML apps. It has four pages: Home explains the project, Data loads the dataset, Train Models trains the two algorithms, and Results shows the performance metrics and confusion matrices."

## 💡 Why This Structure is Better for Freshers

### Before (Complex):
```
src/
├── data/
│   ├── __init__.py
│   ├── data_loader.py (class DataLoader with 10+ methods)
│   ├── preprocessor.py (class DataPreprocessor)
│   └── feature_engineer.py
├── models/
│   ├── base_model.py (abstract base class)
│   ├── logistic_model.py (inherits from BaseModel)
│   ├── random_forest_model.py (inherits from BaseModel)
├── evaluation/
    ├── metrics.py
    └── visualizer.py
```

### After (Simple):
```
src/
├── data_loader.py (2 simple functions)
├── preprocessor.py (2 simple functions)
├── model_trainer.py (3 simple functions)
├── evaluator.py (4 simple functions)
└── app.py (Streamlit dashboard)
```

**Benefits:**
- ✅ No abstract classes to explain
- ✅ No complex inheritance
- ✅ Each file has 2-4 functions max
- ✅ Easy to understand flow
- ✅ No nested imports
- ✅ Flat structure - everything at one level

## 🎓 Complete Example for Interviews

```python
# This is the complete workflow you can show in an interview

# Step 1: Load data
from src.data_loader import load_sample_data
df = load_sample_data()

# Step 2: Prepare data
from src.preprocessor import prepare_data, preprocess_features
X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2)
X_train, X_test = preprocess_features(X_train, X_test)

# Step 3: Train models
from src.model_trainer import train_random_forest, make_predictions
model = train_random_forest(X_train, y_train)

# Step 4: Evaluate
from src.evaluator import calculate_metrics
y_pred, y_proba = make_predictions(model, X_test)
metrics = calculate_metrics(y_test, y_pred, y_proba)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

## 🏃 Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# Run the dashboard
streamlit run streamlit_app.py
```

That's it! The code is now simple, clean, and perfect for explaining in interviews.
