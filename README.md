# Data-Science_Predictive-Maintenance
A full predictive maintenance project using Python — from raw telemetry data to failure prediction and alert generation.

## ⚙️ Predictive Maintenance — 24h Failure Prediction using Random Forest  

This project develops a complete end-to-end machine learning pipeline for **predicting machine failures within the next 24 hours** using telemetry data from industrial equipment.  
It focuses on data preprocessing, rolling feature generation, and building an interpretable **Random Forest** model with solid evaluation metrics and clean visualization inside Jupyter.

---

### 🚀 Key Features

#### 🧹 Data Preparation
- Loaded and cleaned multiple PdM datasets: telemetry, failures, and machine metadata.  
- Handled duplicates, sorted time-series data by `machineID` and `datetime`.  
- Created 24-hour ahead failure labels using **as-of merge** for each machine timeline.

#### 🧮 Feature Engineering
- Computed rolling statistics (mean, std) for key sensors — *voltage, rotation, pressure, and vibration* — over **3H** and **6H** time windows.  
- Integrated machine-level data such as `age` and `model` via one-hot encoding.  
- Scaled continuous features using **StandardScaler** for consistent model behavior.

#### 🤖 Modeling
- Trained a **RandomForestClassifier** with balanced class weights to handle class imbalance.  
- Split data into 80% training / 20% testing for fair evaluation.  
- Implemented clean notebook workflow (no warnings, direct inline plots).

#### 📊 Evaluation
- Metrics: **ROC-AUC, PR-AUC, Precision, Recall, F1-score**, and Confusion Matrix.  
- Visualized **ROC Curve** and **Precision–Recall Curve** to interpret trade-offs between false alarms and missed failures.  

---

### 📈 Results Summary

| Metric | Value | Description |
|:--------|:------|:-------------|
| **ROC-AUC** | 0.964 | Excellent class separation between healthy and failing machines. |
| **PR-AUC** | 0.337 | Moderate precision-recall trade-off under strong class imbalance. |
| **Precision (failure)** | 0.75 | 75% of predicted failures were correct. |
| **Recall (failure)** | 0.04 | Conservative model; few false alarms but low sensitivity. |
| **Accuracy** | 0.981 | High overall accuracy, dominated by healthy class. |

---

### 🧠 Tech Stack

**Language:** Python 3.10+  
**Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Joblib  
**Environment:** Jupyter Notebook 

---

### 💾 Output

- Cleaned and labeled datasets: `tele_labeled`, `tele_feats`  
- Trained model: `rf_model_notebook.joblib` (includes model, scaler, feature names, threshold)  
- Evaluation plots: ROC & PR curves displayed inline in notebook

---

### 📚 Author  

📧 [sina.firuzian@gmail.com] 
🌐 [https://github.com/yourusername]

