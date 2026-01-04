"""
Minimal Streamlit Dashboard - Works without ML dependencies

This is a simplified version that demonstrates the UI without requiring
scikit-learn, tensorflow, etc. Perfect for showcasing the project structure.

Run with: streamlit run streamlit_minimal.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np

# Try to import streamlit, provide helpful error if not available
try:
    import streamlit as st
except ImportError:
    print("❌ Streamlit not installed.")
    print("\nTo install (requires Python 3.11-3.14):")
    print("  pip install streamlit")
    print("\nOr run the simple demo instead:")
    print("  python demo_simple.py")
    sys.exit(1)

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title
st.markdown("# 🧠 Customer Churn Prediction System")
st.markdown("### Production-Ready ML Dashboard (Minimal Demo)")
st.markdown("---")

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📁 Data Explorer", "📚 Documentation"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About This Demo**

This is a minimal version that works without ML dependencies.

For full functionality, use Python 3.11 or 3.12.
""")

# Home Page
if page == "🏠 Home":
    st.header("Welcome to the Customer Churn Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Business Problem
        - Identify at-risk customers
        - Optimize retention campaigns
        - Reduce acquisition costs
        - Increase customer lifetime value
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 ML Models
        - Logistic Regression
        - Random Forest
        - XGBoost
        - Neural Network
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Features
        - Data exploration
        - Model training
        - Performance evaluation
        - Batch predictions
        """)
    
    st.markdown("---")
    
    st.subheader("📈 Project Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Lines of Code", "3,500+")
    col2.metric("ML Models", "4")
    col3.metric("Modules", "20+")
    col4.metric("Sample Data", "7,043 rows")
    
    st.markdown("---")
    
    st.subheader("⚠️ Python 3.15 Compatibility Note")
    st.warning("""
    **Current Limitation**: Python 3.15 is too new for many ML packages.
    
    **Solution**: Use Python 3.11 or 3.12 for full functionality:
    ```bash
    brew install python@3.11
    python3.11 -m venv venv_py311
    source venv_py311/bin/activate
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```
    """)

# Data Explorer Page
elif page == "📁 Data Explorer":
    st.header("📁 Data Explorer")
    
    st.subheader("Load Sample Dataset")
    
    if st.button("Load Sample Churn Data"):
        try:
            from src.data.data_loader import load_sample_data
            df = load_sample_data()
            
            st.success(f"✅ Loaded {len(df)} customers with {len(df.columns)} features")
            
            # Statistics
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df))
            col2.metric("Total Columns", len(df.columns))
            col3.metric("Missing Values", df.isnull().sum().sum())
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(20))
            
            # Basic statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
            # Churn distribution
            if 'Churn' in df.columns:
                st.subheader("Churn Distribution")
                churn_counts = df['Churn'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(churn_counts)
                with col2:
                    st.write("**Counts:**")
                    st.write(churn_counts)
                    churn_rate = (churn_counts.get('Yes', 0) / len(df)) * 100
                    st.metric("Churn Rate", f"{churn_rate:.2f}%")
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values
            })
            st.dataframe(col_info)
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Make sure the sample dataset exists. Run: `python src/data/generate_sample_data.py`")

# Documentation Page
elif page == "📚 Documentation":
    st.header("📚 Project Documentation")
    
    st.subheader("🎯 Project Structure")
    st.code("""
project2/
├── data/
│   └── sample_churn_data.csv      # 7,043 customer records
├── src/
│   ├── config.py                  # Configuration
│   ├── data/                      # Data pipeline
│   ├── models/                    # ML models
│   ├── evaluation/                # Metrics & viz
│   └── utils/                     # Utilities
├── streamlit_app.py               # Full dashboard
├── streamlit_minimal.py           # This minimal version
├── demo_simple.py                 # CLI demo
├── README.md                      # Documentation
└── requirements.txt               # Dependencies
    """, language="text")
    
    st.subheader("🚀 Quick Start")
    st.markdown("""
    **Option 1: Full Installation (Python 3.11/3.12)**
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    streamlit run streamlit_app.py
    ```
    
    **Option 2: Simple Demo (Current Python)**
    ```bash
    python demo_simple.py
    ```
    
    **Option 3: This Minimal Dashboard**
    ```bash
    pip install streamlit pandas
    streamlit run streamlit_minimal.py
    ```
    """)
    
    st.subheader("📊 ML Models Implemented")
    
    models_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"],
        "Type": ["Linear", "Ensemble", "Gradient Boosting", "Deep Learning"],
        "Expected Accuracy": ["~78%", "~82%", "~84%", "~83%"],
        "Training Time": ["~1s", "~5s", "~10s", "~30s"]
    }
    st.table(pd.DataFrame(models_data))
    
    st.subheader("📁 Key Files")
    st.markdown("""
    - **README.md** - Comprehensive project documentation
    - **QUICKSTART.md** - Quick start guide
    - **INSTALL.md** - Installation troubleshooting
    - **walkthrough.md** - Complete project walkthrough
    - **demo_simple.py** - Simple CLI demo
    """)
    
    st.subheader("💼 Resume Value")
    st.success("""
    This project demonstrates:
    - ✅ End-to-end ML workflow
    - ✅ Clean, production-quality code (3,500+ lines)
    - ✅ Multiple ML algorithms
    - ✅ Real business problem
    - ✅ Interactive dashboard
    - ✅ Comprehensive documentation
    
    **Perfect for:** Technical interviews, GitHub portfolio, ML Engineer roles
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Customer Churn Prediction System | Built with ❤️ using Python, scikit-learn, and Streamlit</p>
    <p>© 2024 | Production-Ready ML Dashboard</p>
</div>
""", unsafe_allow_html=True)
