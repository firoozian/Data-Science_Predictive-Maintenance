⚙️ Predictive Maintenance — 24h Failure Prediction using Random Forest
This project develops a complete end-to-end machine learning pipeline for predicting machine failures within the next 24 hours using telemetry data from industrial equipment. It focuses on data preprocessing, rolling feature generation, and building an interpretable Random Forest model with solid evaluation metrics and clean visualization inside Jupyter.

🚀 Key Features  
🧹 Data Preparation  
Loaded and cleaned multiple PdM datasets: telemetry, failures, errors, maintenance logs, and machine metadata.  
Handled duplicates, sorted time-series data by machineID and datetime.  
Created 24-hour ahead failure labels using as-of merge for each machine timeline.

🧮 Feature Engineering  
Computed rolling statistics (mean, std) for key sensors — voltage, rotation, pressure, and vibration — over 3H and 6H time windows.  
Integrated machine-level data such as age and model via one-hot encoding.  
Scaled continuous features using StandardScaler for consistent model behavior.

🤖 Modeling  
Trained a RandomForestClassifier with balanced class weights to handle class imbalance.  
Split data into 80% training / 20% testing.  
Clean notebook workflow (no warnings, direct inline plots).

📊 Evaluation  
Metrics: ROC-AUC, PR-AUC, Precision, Recall, F1-score, and Confusion Matrix.  
Visualized ROC Curve and Precision–Recall Curve to understand trade-offs.

📈 Results Summary (Updated)
| Metric | Value | Description |
|-------|-------|-------------|
| **ROC-AUC** | **0.9982** | Excellent class separation |
| **PR-AUC** | **0.9031** | Strong performance under class imbalance |
| **Precision (failure)** | **0.848** | 84.8% of predicted failures were correct |
| **Recall (failure)** | **0.822** | Successfully detects majority of failures |
| **F1-score (failure)** | **0.835** | Balanced precision–recall performance |
| **Accuracy** | **0.994** | High overall accuracy across all machines |

**Confusion Matrix:**
[[68357   198]  
 [  240  1105]]

🔥 Most Important Features  
- `error_count_24h`  
- `rotate_mean_6H`, `volt_mean_6H`, `vibration_mean_6H`, `pressure_mean_6H`  
- `rotate_mean_3H`, `vibration_mean_3H`, `volt_mean_3H`, `pressure_mean_3H`  

🧠 Tech Stack  
Python 3.10+ • Pandas • NumPy • Scikit-learn • Matplotlib • Joblib  
Environment: Jupyter Notebook

💾 Output  
Cleaned and labeled datasets: `tele_labeled`, `tele_feats`  
Trained model artifact: `rf_model_notebook.joblib` (model + scaler + feature names + threshold)  
Evaluation plots displayed inline


📚 Author  

📧 [sina.firuzian@gmail.com] 

