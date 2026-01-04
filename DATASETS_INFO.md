# 📊 Available Datasets

## 1. Full Sample Dataset (Recommended)
**File**: `data/sample_churn_data.csv`  
**Size**: 7,043 customers  
**Features**: 21 columns  
**Churn Rate**: 56.41%

### Features Include:
- **Demographics**: Gender, SeniorCitizen, Partner, Dependents
- **Account Info**: Tenure, Contract type, Payment method, Billing
- **Services**: Phone, Internet, Streaming, Security, Support
- **Charges**: Monthly charges, Total charges
- **Target**: Churn (Yes/No)

### Use This For:
✅ Training all ML models  
✅ Comprehensive model evaluation  
✅ Feature importance analysis  
✅ Production-ready results

---

## 2. Small Test Dataset (Quick Testing)
**File**: `data/test_sample_small.csv`  
**Size**: 20 customers  
**Features**: 21 columns (same as full dataset)  
**Churn Rate**: 50%

### Use This For:
✅ Quick model testing  
✅ Debugging  
✅ Understanding data structure  
✅ Fast iterations

---

## 📁 Dataset Locations

```
project2/
├── data/
│   ├── sample_churn_data.csv      # Full dataset (7,043 rows)
│   └── test_sample_small.csv      # Small test (20 rows)
```

---

## 🚀 How to Use in Dashboard

### Option 1: Load Sample Dataset (Full)
1. Go to **📁 Data Explorer** page
2. Select **"Sample Dataset"** radio button
3. Click **"Load Sample Churn Dataset"** button
4. View 7,043 customers with full statistics

### Option 2: Upload Custom CSV (Small Test)
1. Go to **📁 Data Explorer** page
2. Select **"Upload CSV"** radio button
3. Click **"Browse files"** and select `data/test_sample_small.csv`
4. View 20 customers for quick testing

---

## 🤖 Training Models

Once data is loaded:

1. **Go to "🤖 Train Models"** page
2. **Prepare Data**:
   - Set test size (default: 20%)
   - Enable feature engineering (recommended)
   - Click "Prepare Data"

3. **Select Models** to train:
   - ☑️ Logistic Regression (fast, baseline)
   - ☑️ Random Forest (good performance)
   - ☑️ XGBoost (best accuracy)
   - ☑️ Neural Network (deep learning)

4. **Click "🚀 Train Selected Models"**

5. **View Results** immediately after training

---

## 📈 Expected Performance

### On Full Dataset (7,043 customers):

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Logistic Regression | ~78% | ~1 second |
| Random Forest | ~82% | ~5 seconds |
| **XGBoost** | **~84%** | ~10 seconds |
| Neural Network | ~83% | ~30 seconds |

### On Small Dataset (20 customers):
- **Note**: Results will be less reliable due to small sample size
- Use for testing workflow, not for actual predictions
- Training will be very fast (< 1 second per model)

---

## 💡 Tips

1. **Start with Full Dataset**: Better results and more reliable metrics
2. **Use Small Dataset**: Only for quick testing or debugging
3. **Feature Engineering**: Always enable for better performance
4. **Model Comparison**: Train multiple models to compare performance
5. **Save Models**: Models are automatically saved after training

---

## 📝 Dataset Schema

```
customerID         : Unique customer identifier
gender            : Male/Female
SeniorCitizen     : 0 (No) or 1 (Yes)
Partner           : Yes/No
Dependents        : Yes/No
tenure            : Months with company (0-72)
PhoneService      : Yes/No
MultipleLines     : Yes/No/No phone service
InternetService   : DSL/Fiber optic/No
OnlineSecurity    : Yes/No/No internet service
OnlineBackup      : Yes/No/No internet service
DeviceProtection  : Yes/No/No internet service
TechSupport       : Yes/No/No internet service
StreamingTV       : Yes/No/No internet service
StreamingMovies   : Yes/No/No internet service
Contract          : Month-to-month/One year/Two year
PaperlessBilling  : Yes/No
PaymentMethod     : Electronic check/Mailed check/Bank transfer/Credit card
MonthlyCharges    : Monthly bill amount ($)
TotalCharges      : Total amount charged ($)
Churn             : Yes/No (TARGET VARIABLE)
```

---

**Ready to train models!** 🚀
