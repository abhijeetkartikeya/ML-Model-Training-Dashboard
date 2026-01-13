"""
Streamlit App for Customer Churn Prediction

Simple dashboard for training and evaluating churn prediction models.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import our simple modules
from src.data_loader import load_sample_data
from src.preprocessor import prepare_data, preprocess_features
from src.model_trainer import (
    train_logistic_regression,
    train_random_forest,
    make_predictions
)
from src.evaluator import (
    calculate_metrics,
    plot_confusion_matrix,
    get_feature_importance,
    plot_feature_importance
)


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧠 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Simple ML Dashboard for Predicting Customer Churn</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📁 Data", "🤖 Train Models", "📈 Results"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About**

This app predicts customer churn using:
- Logistic Regression
- Random Forest

Simple and easy to understand!
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
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}


# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.header("Welcome!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What is Customer Churn?
        Customer churn is when customers stop using a service.
        
        **Why predict it?**
        - Save customers before they leave
        - Reduce marketing costs
        - Increase revenue
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Our Models
        We use two simple models:
        
        1. **Logistic Regression** - Fast and simple
        2. **Random Forest** - More accurate
        """)
    
    st.markdown("---")
    
    st.subheader("📝 How to Use")
    st.markdown("""
    1. Go to **📁 Data** to load the dataset
    2. Go to **🤖 Train Models** to train machine learning models  
    3. Go to **📈 Results** to see how well the models work
    """)


# ==================== DATA PAGE ====================
elif page == "📁 Data":
    st.header("📁 Load Data")
    
    if st.button("Load Sample Dataset"):
        try:
            st.session_state.data = load_sample_data()
            st.success(f"✅ Loaded {len(st.session_state.data)} customers!")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure you have generated the sample data first.")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Features", len(df.columns))
        col3.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10))
        
        st.subheader("Churn Distribution")
        if 'Churn' in df.columns:
            churn_counts = df['Churn'].value_counts()
            st.bar_chart(churn_counts)


# ==================== TRAIN MODELS PAGE ====================
elif page == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load data first!")
    else:
        st.subheader("Step 1: Prepare Data")
        
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        
        if st.button("Prepare Data"):
            with st.spinner("Preparing data..."):
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = prepare_data(
                        st.session_state.data,
                        test_size=test_size,
                        drop_columns=['customerID'] if 'customerID' in st.session_state.data.columns else None
                    )
                    
                    # Preprocess
                    X_train_processed, X_test_processed = preprocess_features(X_train, X_test)
                    
                    #Store in session
                    st.session_state.X_train = X_train_processed
                    st.session_state.X_test = X_test_processed
                    st.session_state.y_train = y_train.values
                    st.session_state.y_test = y_test.values
                    
                    st.success("✅ Data prepared successfully!")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        
        st.subheader("Step 2: Train Models")
        
        if st.session_state.X_train is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Train Logistic Regression"):
                    with st.spinner("Training..."):
                        try:
                            # Train model
                            model = train_logistic_regression(
                                st.session_state.X_train,
                                st.session_state.y_train
                            )
                            
                            # Make predictions
                            y_pred, y_proba = make_predictions(model, st.session_state.X_test)
                            
                            # Calculate metrics
                            metrics = calculate_metrics(
                                st.session_state.y_test,
                                y_pred
                            )
                            
                            # Store results
                            st.session_state.models['Logistic Regression'] = model
                            st.session_state.results['Logistic Regression'] = {
                                'y_pred': y_pred,
                                'y_proba': y_proba,
                                'metrics': metrics
                            }
                            
                            st.success("✅ Logistic Regression trained!")
                            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                if st.button("Train Random Forest"):
                    with st.spinner("Training..."):
                        try:
                            # Train model
                            model = train_random_forest(
                                st.session_state.X_train,
                                st.session_state.y_train
                            )
                            
                            # Make predictions
                            y_pred, y_proba = make_predictions(model, st.session_state.X_test)
                            
                            # Calculate metrics
                            metrics = calculate_metrics(
                                st.session_state.y_test,
                                y_pred
                            )
                            
                            # Store results
                            st.session_state.models['Random Forest'] = model
                            st.session_state.results['Random Forest'] = {
                                'y_pred': y_pred,
                                'y_proba': y_proba,
                                'metrics': metrics
                            }
                            
                            st.success("✅ Random Forest trained!")
                            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("Please prepare data first!")


# ==================== RESULTS PAGE ====================
elif page == "📈 Results":
    st.header("📈 Model Results")
    
    if len(st.session_state.results) == 0:
        st.warning("⚠️ No trained models yet. Train some models first!")
    else:
        # Model comparison
        st.subheader("Model Comparison")
        
        comparison_data = []
        for model_name, result in st.session_state.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        st.markdown("---")
        
        # Detailed results for each model
        st.subheader("Detailed Results")
        
        selected_model = st.selectbox("Select model:", list(st.session_state.results.keys()))
        
        if selected_model:
            result = st.session_state.results[selected_model]
            metrics = result['metrics']
            
            # Show metrics
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            fig = plot_confusion_matrix(metrics['confusion_matrix'], title=f"{selected_model} - Confusion Matrix")
            st.pyplot(fig)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Customer Churn Prediction | Built with Streamlit & scikit-learn</p>
</div>
""", unsafe_allow_html=True)
