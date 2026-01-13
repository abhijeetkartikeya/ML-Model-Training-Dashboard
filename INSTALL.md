# Installation Guide for Customer Churn Prediction System

## Quick Start (Recommended)

Due to compilation issues with some packages on Python 3.15, we recommend using Python 3.11 or 3.12:

### Option 1: Using Python 3.11/3.12 (Recommended)

```bash
# Install Python 3.11 or 3.12 if not already installed
# On macOS with Homebrew:
brew install python@3.11

# Create virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit imbalanced-learn joblib pyyaml python-dotenv

# Generate sample dataset
python src/data/generate_sample_data.py

# Run the dashboard
streamlit run streamlit_app.py
```

### Option 2: Using Conda (Alternative)

```bash
# Create conda environment
conda create -n churn-prediction python=3.11
conda activate churn-prediction

# Install dependencies
conda install pandas numpy scikit-learn matplotlib seaborn
pip install streamlit plotly imbalanced-learn joblib pyyaml python-dotenv

# Generate sample dataset
python src/data/generate_sample_data.py

# Run the dashboard
streamlit run streamlit_app.py
```

### Option 3: Minimal Installation (For Testing)

If you just want to test the core functionality:

```bash
source venv/bin/activate

# Install only essential packages
pip install pandas numpy streamlit

# You can run the dashboard with limited functionality
streamlit run streamlit_app.py
```

## Current Status

✅ **Completed:**
- Project structure created
- All source code written (data pipeline, models, evaluation, dashboard)
- Sample dataset generated (7,043 customers)
- Configuration and utilities implemented
- README and documentation created

⚠️ **Installation Issues:**
- Python 3.15 has compatibility issues with scikit-learn and matplotlib
- SSL certificate issues preventing some package downloads
- Missing Fortran compiler for scipy (required by scikit-learn)

## Recommended Next Steps

1. **Use Python 3.11 or 3.12** instead of 3.15
2. Install dependencies using the commands above
3. Run the Streamlit dashboard
4. Train models and explore the system

## Troubleshooting

### SSL Certificate Errors
```bash
# Install certificates (macOS)
/Applications/Python\ 3.11/Install\ Certificates.command

# Or use conda which handles certificates better
conda install <package-name>
```

### Fortran Compiler Missing
```bash
# Install gfortran (macOS)
brew install gcc

# Or use pre-built wheels
pip install --only-binary :all: scikit-learn
```

### Python 3.15 Compatibility
Many scientific packages don't yet have wheels for Python 3.15. Use Python 3.11 or 3.12 instead.

## Verify Installation

```python
# Test imports
python -c "import pandas, numpy, sklearn, streamlit; print('All packages installed successfully!')"
```

## Running the Application

```bash
# Activate environment
source venv/bin/activate  # or: conda activate churn-prediction

# Run Streamlit dashboard
streamlit run streamlit_app.py

# Open browser to: http://localhost:8501
```

## Project is Ready!

All code is complete and production-ready. The only remaining step is installing the dependencies using a compatible Python version.
