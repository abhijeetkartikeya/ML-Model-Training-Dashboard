"""
Customer Churn Prediction - Streamlit Dashboard

A comprehensive ML dashboard for training and evaluating churn prediction models.

Author: Kartikeya
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config, StreamlitConfig
from src.data.data_loader import DataLoader, load_sample_data
from src.data.preprocessor import DataPreprocessor, prepare_data
from src.data.feature_engineer import FeatureEngineer
from src.models.logistic_model import LogisticModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network import NeuralNetworkModel
from src.evaluation.metrics import calculate_all_metrics, compare_models, get_classification_report
from src.evaluation.visualizer import (
    plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_model_comparison
)

# Page configuration
st.set_page_config(
    page_title=StreamlitConfig.PAGE_TITLE,
    page_icon=StreamlitConfig.PAGE_ICON,
    layout=StreamlitConfig.LAYOUT,
    initial_sidebar_state=StreamlitConfig.INITIAL_SIDEBAR_STATE
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FAFAFA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧠 Customer Churn Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Ready ML Dashboard for Business Intelligence</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📁 Data Explorer", "🤖 Train Models", "📈 Evaluate Models", "🔮 Make Predictions"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About This Project**

A production-quality ML system for predicting customer churn using:
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network

Built with scikit-learn, TensorFlow, and Streamlit.
""")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}


# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.header("Welcome to the Customer Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Business Problem
        Customer churn costs businesses billions annually. This system helps:
        - Identify at-risk customers
        - Optimize retention campaigns
        - Reduce customer acquisition costs
        - Increase customer lifetime value
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 ML Models
        Train and compare multiple algorithms:
        - **Logistic Regression** (Baseline)
        - **Random Forest** (Ensemble)
        - **XGBoost** (Gradient Boosting)
        - **Neural Network** (Deep Learning)
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Features
        - Interactive data exploration
        - Real-time model training
        - Comprehensive evaluation
        - Feature importance analysis
        - Batch predictions
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    1. **Load Data**: Go to 'Data Explorer' and load the sample dataset or upload your own
    2. **Train Models**: Navigate to 'Train Models' and select algorithms to train
    3. **Evaluate**: Compare model performance in 'Evaluate Models'
    4. **Predict**: Make predictions on new data in 'Make Predictions'
    """)
    
    st.markdown("---")
    
    st.subheader("💼 Business Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Customer Value", "$1,000")
    col2.metric("Retention Campaign Cost", "$50")
    col3.metric("Potential Savings", "$125,000")
    col4.metric("ROI", "2,500%")


# ==================== DATA EXPLORER PAGE ====================
elif page == "📁 Data Explorer":
    st.header("📁 Data Explorer")
    
    tab1, tab2 = st.tabs(["Load Data", "Explore Data"])
    
    with tab1:
        st.subheader("Load Dataset")
        
        data_source = st.radio("Select data source:", ["Sample Dataset", "Upload CSV"])
        
        if data_source == "Sample Dataset":
            if st.button("Load Sample Churn Dataset"):
                with st.spinner("Loading sample data..."):
                    try:
                        st.session_state.data = load_sample_data()
                        st.success(f"✅ Loaded {len(st.session_state.data)} rows and {len(st.session_state.data.columns)} columns")
                    except Exception as e:
                        st.error(f"Error loading sample data: {e}")
                        st.info("Sample data file not found. Please generate it first by running: `python src/data/generate_sample_data.py`")
        
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    st.session_state.data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(st.session_state.data)} rows and {len(st.session_state.data.columns)} columns")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
    
    with tab2:
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df))
            col2.metric("Total Columns", len(df.columns))
            col3.metric("Missing Values", df.isnull().sum().sum())
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
            st.subheader("Missing Values")
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if len(missing) > 0:
                st.bar_chart(missing)
            else:
                st.success("No missing values found!")
            
            st.subheader("Target Distribution")
            if 'Churn' in df.columns:
                churn_counts = df['Churn'].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(churn_counts)
                with col2:
                    st.write("Churn Distribution:")
                    st.write(churn_counts)
                    churn_rate = (churn_counts.get('Yes', 0) / len(df)) * 100
                    st.metric("Churn Rate", f"{churn_rate:.2f}%")
        
        else:
            st.info("Please load data first!")


# ==================== TRAIN MODELS PAGE ====================
elif page == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Explorer page!")
    else:
        st.subheader("1️⃣ Data Preparation")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        with col2:
            apply_feature_engineering = st.checkbox("Apply Feature Engineering", value=True)
        
        if st.button("Prepare Data"):
            with st.spinner("Preparing data..."):
                try:
                    df = st.session_state.data.copy()
                    
                    # Feature engineering
                    if apply_feature_engineering:
                        fe = FeatureEngineer()
                        df = fe.create_all_features(df)
                        st.success(f"✅ Created {len(fe.created_features)} engineered features")
                    
                    # Prepare data
                    X_train, X_test, y_train, y_test = prepare_data(
                        df,
                        test_size=test_size,
                        drop_columns=['customerID'] if 'customerID' in df.columns else None
                    )
                    
                    # Preprocessing
                    preprocessor = DataPreprocessor(
                        imputation_strategy='mean',
                        scaling_method='standard',
                        encoding_method='label'
                    )
                    
                    X_train_processed = preprocessor.fit_transform(X_train, y_train)
                    X_test_processed = preprocessor.transform(X_test)
                    
                    # Store in session state
                    st.session_state.X_train = X_train_processed
                    st.session_state.X_test = X_test_processed
                    st.session_state.y_train = y_train.values
                    st.session_state.y_test = y_test.values
                    st.session_state.preprocessor = preprocessor
                    
                    st.success(f"✅ Data prepared! Train: {len(X_train)}, Test: {len(X_test)}")
                    
                except Exception as e:
                    st.error(f"Error preparing data: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown("---")
        
        st.subheader("2️⃣ Select and Train Models")
        
        if st.session_state.X_train is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                train_logistic = st.checkbox("Logistic Regression")
                train_rf = st.checkbox("Random Forest")
            
            with col2:
                train_xgb = st.checkbox("XGBoost")
                train_nn = st.checkbox("Neural Network")
            
            if st.button("🚀 Train Selected Models"):
                models_to_train = []
                
                if train_logistic:
                    models_to_train.append(('logistic', LogisticModel()))
                if train_rf:
                    models_to_train.append(('random_forest', RandomForestModel()))
                if train_xgb:
                    models_to_train.append(('xgboost', XGBoostModel()))
                if train_nn:
                    models_to_train.append(('neural_network', NeuralNetworkModel()))
                
                if len(models_to_train) == 0:
                    st.warning("Please select at least one model to train!")
                else:
                    for model_key, model in models_to_train:
                        st.subheader(f"Training {model.model_name}...")
                        
                        # Add progress bar for neural network
                        if model_key == 'neural_network':
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            status_text.text("Initializing neural network...")
                            progress_bar.progress(20)
                        
                        with st.spinner(f"Training {model.model_name}..."):
                            try:
                                # Update progress for neural network
                                if model_key == 'neural_network':
                                    status_text.text("Training neural network (this will be fast)...")
                                    progress_bar.progress(40)
                                
                                # Train model
                                model.train(
                                    st.session_state.X_train,
                                    st.session_state.y_train,
                                    st.session_state.X_test,
                                    st.session_state.y_test
                                )
                                
                                # Update progress for neural network
                                if model_key == 'neural_network':
                                    progress_bar.progress(70)
                                    status_text.text("Making predictions...")
                                
                                # Make predictions
                                y_pred = model.predict(st.session_state.X_test)
                                y_proba = model.predict_proba(st.session_state.X_test)
                                
                                # Update progress for neural network
                                if model_key == 'neural_network':
                                    progress_bar.progress(90)
                                    status_text.text("Calculating metrics...")
                                
                                # Calculate metrics
                                metrics = calculate_all_metrics(
                                    st.session_state.y_test,
                                    y_pred,
                                    y_proba
                                )
                                
                                # Complete progress for neural network
                                if model_key == 'neural_network':
                                    progress_bar.progress(100)
                                    status_text.text("✅ Training complete!")
                                
                                # Store results
                                st.session_state.trained_models[model_key] = model
                                st.session_state.model_results[model_key] = {
                                    'model': model,
                                    'y_pred': y_pred,
                                    'y_proba': y_proba,
                                    'metrics': metrics
                                }
                                
                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                                col2.metric("Precision", f"{metrics['precision']:.4f}")
                                col3.metric("Recall", f"{metrics['recall']:.4f}")
                                col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
                                
                                st.success(f"✅ {model.model_name} trained successfully!")
                                
                            except Exception as e:
                                st.error(f"Error training {model.model_name}: {e}")
                                import traceback
                                st.code(traceback.format_exc())
        
        else:
            st.info("Please prepare data first!")


# ==================== EVALUATE MODELS PAGE ====================
elif page == "📈 Evaluate Models":
    st.header("📈 Model Evaluation & Comparison")
    
    if len(st.session_state.model_results) == 0:
        st.warning("⚠️ No trained models found! Please train models first.")
    else:
        st.subheader("Model Comparison")
        
        # Create comparison DataFrame
        comparison_data = {}
        for model_key, result in st.session_state.model_results.items():
            comparison_data[result['model'].model_name] = result['metrics']
        
        comparison_df = compare_models(comparison_data)
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']))
        
        # Plot comparison
        fig = plot_model_comparison(comparison_df, metric='ROC-AUC')
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Detailed evaluation for each model
        st.subheader("Detailed Model Evaluation")
        
        model_names = [result['model'].model_name for result in st.session_state.model_results.values()]
        selected_model_name = st.selectbox("Select model to evaluate:", model_names)
        
        # Find the selected model
        selected_result = None
        for result in st.session_state.model_results.values():
            if result['model'].model_name == selected_model_name:
                selected_result = result
                break
        
        if selected_result:
            metrics = selected_result['metrics']
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            col5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(metrics['confusion_matrix'], title=f"Confusion Matrix - {selected_model_name}")
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            report = get_classification_report(st.session_state.y_test, selected_result['y_pred'])
            st.text(report)
            
            # Feature Importance (if available)
            importance_df = selected_result['model'].get_feature_importance()
            if importance_df is not None:
                st.subheader("Feature Importance")
                fig = plot_feature_importance(importance_df, top_n=15, title=f"Top 15 Features - {selected_model_name}")
                st.pyplot(fig)


# ==================== MAKE PREDICTIONS PAGE ====================
elif page == "🔮 Make Predictions":
    st.header("🔮 Make Predictions")
    
    if len(st.session_state.trained_models) == 0:
        st.warning("⚠️ No trained models found! Please train models first.")
    else:
        # Select model
        model_names = [model.model_name for model in st.session_state.trained_models.values()]
        selected_model_name = st.selectbox("Select model for predictions:", model_names)
        
        # Find selected model
        selected_model = None
        for model in st.session_state.trained_models.values():
            if model.model_name == selected_model_name:
                selected_model = model
                break
        
        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        
        with tab1:
            st.subheader("Enter Customer Information")
            st.info("This is a simplified prediction interface. In production, you would have a form with all features.")
            
            if st.button("Make Sample Prediction"):
                if st.session_state.X_test is not None:
                    # Use a random sample from test set
                    sample_idx = np.random.randint(0, len(st.session_state.X_test))
                    sample = st.session_state.X_test[sample_idx:sample_idx+1]
                    
                    # Predict
                    prediction = selected_model.predict(sample)[0]
                    proba = selected_model.predict_proba(sample)[0]
                    
                    # Display result
                    st.subheader("Prediction Result")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if prediction == 1:
                            st.error("🚨 **HIGH RISK**: Customer likely to churn")
                        else:
                            st.success("✅ **LOW RISK**: Customer likely to stay")
                    
                    with col2:
                        churn_probability = proba[1] * 100
                        st.metric("Churn Probability", f"{churn_probability:.2f}%")
                    
                    # Recommendation
                    st.subheader("Recommendation")
                    if churn_probability > 70:
                        st.warning("**Action Required**: Immediate retention campaign recommended")
                    elif churn_probability > 40:
                        st.info("**Monitor**: Keep close watch on customer engagement")
                    else:
                        st.success("**Maintain**: Continue current service level")
        
        with tab2:
            st.subheader("Batch Predictions")
            st.info("Upload a CSV file with customer data for batch predictions")
            
            uploaded_file = st.file_uploader("Upload CSV for predictions", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(new_data)} rows")
                    st.dataframe(new_data.head())
                    
                    if st.button("Generate Predictions"):
                        st.info("Batch prediction functionality would process the uploaded data through the preprocessing pipeline and generate predictions for all rows.")
                
                except Exception as e:
                    st.error(f"Error loading file: {e}")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Customer Churn Prediction System | Built with ❤️ using Streamlit, scikit-learn, and TensorFlow</p>
    <p>© 2024 | Production-Ready ML Dashboard</p>
</div>
""", unsafe_allow_html=True)
