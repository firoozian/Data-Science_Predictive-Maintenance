# ⚙️ Predictive Maintenance — 24h Failure Prediction using Random Forest + Streamlit Dashboard

This project builds a **complete end-to-end machine learning pipeline** for predicting machine failures within the next 24 hours using telemetry data from industrial equipment.  
It includes **data preprocessing, feature engineering, model training, evaluation, and interactive visualization via a Streamlit web app.**

---

## 🚀 Key Features

### 🧹 Data Preparation
- Loaded and cleaned multiple PdM datasets: telemetry, failures, maintenance logs, errors, and machine metadata.  
- Removed duplicates and sorted records by `machineID` and `datetime`.  
- Generated 24-hour-ahead failure labels using a **forward as-of merge** per machine timeline.  

---

### 🧮 Feature Engineering
- Computed **rolling statistics (mean, std)** for key sensors (`volt`, `rotate`, `pressure`, `vibration`) over 3H and 6H windows.  
- Added maintenance and error-based features:
  - `last_maint_component`  
  - `error_count_24h` (number of errors in the last 24 hours)  
- Integrated machine metadata (`age`, `model`) via one-hot encoding.  
- Standardized all numerical features using **StandardScaler** for consistent model behavior.  

---

### 🤖 Modeling
- Trained a **RandomForestClassifier** with:
  - `n_estimators=300`  
  - `class_weight="balanced_subsample"`  
  - `min_samples_split=4`  
- Data split: 80% train / 20% test (stratified).  
- Saved all trained artifacts (model, scaler, columns, threshold) for deployment.  

---

### 📊 Evaluation
- Metrics:
  - **ROC-AUC**, **PR-AUC**, **Precision**, **Recall**, **F1**, **Accuracy**
- Plots:
  - **ROC Curve**
  - **Precision–Recall Curve**
- **Threshold optimization** using F₂-score (recall-weighted).  

---

## 📈 Results Summary

| Metric | Value | Description |
|:--|:--|:--|
| ROC-AUC | **0.9982** | Excellent class separation |
| PR-AUC | **0.9031** | Strong performance under class imbalance |
| Precision (failure) | **0.848** | 84.8% of predicted failures were correct |
| Recall (failure) | **0.822** | Model detects most upcoming failures |
| F1-score (failure) | **0.835** | Balanced precision–recall trade-off |
| Accuracy | **0.994** | High overall accuracy |
| Confusion Matrix | `[[68357 198], [240 1105]]` | Few false positives/negatives |

---

## Threshold Strategy (F₂-Score & Recall Priority)

In predictive maintenance, missing a true failure (false negative) is far more dangerous and costly than raising a false alert.
To prioritize early failure detection, the model uses recall-focused threshold optimization based on the F₂-score (β = 2).

This shifts the decision threshold to catch more upcoming failures while keeping false alarms at a manageable level.


---

### 🔥 Most Important Features
- `error_count_24h`
- `rotate_mean_6H`, `volt_mean_6H`, `vibration_mean_6H`, `pressure_mean_6H`
- `rotate_mean_3H`, `vibration_mean_3H`, `volt_mean_3H`, `pressure_mean_3H`

---

### 🔗 Live Dashboard (Demo)
https://public.tableau.com/app/profile/sina.firoozian/viz/Book1_17631841428950/Story1

---

## 💻 Streamlit Web App

A fully interactive **Streamlit dashboard** for batch scoring and failure-risk analysis.

### 🧭 App Overview
- **Upload a CSV** with pre-engineered features (same schema as training).  
- **Adjust decision threshold** via slider.  
- **Instant predictions** with failure probabilities and labels.  
- **Download results** as CSV (`predictions_pdm.csv`).  
- Built-in **Analytics Dashboard**:
  - ✅ Summary statistics (total samples, failure rate, avg probability)  
  - 📊 Probability distribution histogram with threshold line  
  - 🥧 Class distribution pie chart  
  - ⚠️ Top-10 highest-risk samples with export option  

---

### 🧩 Artifacts Used by App
All saved in:  
`C:\Probook\E\AI\Projects\Data Science\Predictive Maintenance\predictive_maintenance_app\artifacts`

| File | Purpose |
|------|----------|
| `model_rf.pkl` | Trained Random Forest model |
| `scaler.pkl` | StandardScaler for normalization |
| `feature_columns.json` | Ordered list of feature names |
| `best_threshold.txt` | Optimal decision threshold (F₂) |
| `sample_features.csv` | Example input for testing the app |

---

## 🧠 Tech Stack
**Python 3.10+**  
**Pandas • NumPy • scikit-learn • Matplotlib • Streamlit • Joblib**

**Environment:** Jupyter Notebook → Streamlit App  

---

## 💾 Outputs
- Cleaned datasets: `tele_labeled`, `tele_feats`  
- Trained model and scaler artifacts for deployment  
- Visualization & analytics dashboard via Streamlit
- Exported telemetry_eda.csv and tableau_dataset.csv for Tableau dashboards (sensor analysis and model prediction exploration).
  
---

## 🧑‍💻 Author
**Sina Firuzian**  
📧 [sina.firuzian@gmail.com]  
