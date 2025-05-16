# Employee Burnout Prediction  

**Technical & Performance Report**  
**Audience:** Data Scientists

---

## 1. Project Overview  

**Objective:** Build a machine-learning system to predict an employee’s burnout risk level—categorized as **No Burnout**, **Moderate Risk**, or **Severe Burnout**—based on workplace and personal factors.  
**Motivation:** Burnout leads to decreased productivity, turnover, and health issues. Proactive identification enables targeted interventions and well-being programs.

---

## 2. Data Exploration  

**Dataset:** ~22,750 employee records with features:  

- **EmployeeID** (dropped prior to modeling)  
- **DateOfJoining** (all in late 2008, low variance → unused)  
- **Gender** (Male/Female)  
- **CompanyType** (Service/Product)  
- **WFHSetup** (Yes/No)  
- **Designation** (ordinal 0–5)  
- **ResourceAllocation** (1.0–10.0 workload scale)  
- **MentalFatigueScore** (0.0–10.0)  
- **BurnRate** (0.0–1.0 continuous target)

**Missing Values:**  

- ResourceAllocation ~6% missing  
- MentalFatigueScore ~9% missing  
- BurnRate ~5% missing  

Missingness was largely non-overlapping; rows with missing **BurnRate** were dropped for supervised training, while missing features were imputed (median).

**Key EDA Findings:**  

- **Class Distribution (after bucketing BurnRate):**  
  - No Burnout (< 0.5): ~60%  
  - Moderate (0.5–0.7): ~30%  
  - Severe (> 0.7): ~10%  
- **Strong Correlations:**  
  - MentalFatigue ↔ BurnRate (r ≈ 0.94)  
  - ResourceAllocation ↔ BurnRate (high fatigue & high workload → high burnout)  
- **Weaker Effects:**  
  - Designation showed little trend with burnout.  
  - CompanyType: service-based slightly higher burnout than product.  
  - WFHSetup and Gender had minimal overall impact (minor differences in extremes).

---

## 3. Preprocessing & Feature Engineering  

1. **Dropped Irrelevant Columns:** EmployeeID, DateOfJoining  
2. **Imputation:** Median for ResourceAllocation & MentalFatigue  
3. **Encoding:**  
   - Gender, CompanyType, WFHSetup → binary (0/1)  
   - Designation → ordinal integer (0–5)  
4. **Scaling:**  
   - ResourceAllocation normalized to [0,1] (÷10)  
   - MentalFatigue left on original scale; tree models used so scaling non-critical  
5. **Target Bucketing:**  
   - Continuous BurnRate → 3 classes via thresholds: <0.5, 0.5–0.7, >0.7  
6. **Class Imbalance Handling:**  
   - Stratified splitting and class-weight adjustments or oversampling within CV folds to boost recall on the severe class

---

## 4. Modeling Approach  

- **Algorithms Evaluated:**  
  1. Logistic Regression (baseline)  
  2. Decision Tree  
  3. Random Forest  
  4. XGBoost (**final choice**)  

- **Pipeline:**  
  - Stratified 80/20 train/test split  
  - 5-fold cross-validation on training set, with imputation/encoding/scaling inside CV  
  - Hyperparameter search via GridSearchCV/RandomizedSearchCV  

- **Hyperparameters Tuned:**  
  - **Random Forest:** `n_estimators`, `max_depth`, `class_weight`  
  - **XGBoost:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, regularization (`gamma`, `lambda`)  

- **Rationale for XGBoost:**  
  - Highest overall accuracy (≈88%) and macro-F1 (≈0.80) in CV and test  
  - Superior recall on the rare “Severe” class (~0.78) with acceptable precision (~0.70)  
  - Native handling of missing values and efficient training

---

## 5. Performance Evaluation  

| Metric              | No Burnout | Moderate Risk | Severe Burnout | Macro-Avg |
|---------------------|------------|---------------|----------------|-----------|
| **Precision**       | 0.90       | 0.80          | 0.70           | 0.80      |
| **Recall**          | 0.92       | 0.75          | 0.78           | 0.82      |
| **F1-Score**        | 0.91       | 0.77          | 0.74           | 0.81      |
| **Accuracy (Overall)** | **87%** | –             | –              | –         |

- **Confusion Matrix:** Most misclassifications occur between adjacent classes (moderate ↔ severe).  
- **ROC-AUC (One-vs-Rest):**  
  - Severe vs rest: ~0.88  
  - Moderate vs rest: ~0.85

---

## 6. Deployment Readiness  

- **Model Serialization:** `xgboost_classifier.pkl` via pickle/joblib  
- **Web App Interfaces:**  
  - **Flask** app with HTML form → `/predict` endpoint, color-coded results, Plotly pie chart  
  - **Streamlit** prototype (`app.py`) with sliders & selectboxes for quick demos  
- **Recommendations Engine:** Contextual advice displayed based on predicted class (e.g., self-care tips, workload reduction, professional help)  
- **Environment & Requirements:**  
  - `requirements.txt` lists pandas, numpy, scikit-learn, xgboost, Flask, Plotly, etc.  
  - Hosted demo on PythonAnywhere; instructions for local (`flask run` or `streamlit run app.py`)  
- **Pipeline Maintainability:** Clear separation of training notebooks, model artifacts, and app code enables easy retraining and redeployment

---

## 7. Technical Stack  

- **Data & Modeling:** Python, pandas, NumPy, scikit-learn, XGBoost  
- **Visualization:** Matplotlib, Seaborn (EDA); Plotly (web charts);
- **Web Frameworks:** Flask (production-style app), Streamlit (prototype/demo)  
- **Deployment:** Streamlit, GitHub for version control  
- **Utilities:** joblib/pickle (model persistence), python-dateutil/pytz (date handling)

---

## 8. Potential Improvement Areas  

1. **Feature Enrichment:**  
   - Add tenure, department, performance ratings, overtime hours  
   - Incorporate external factors (team size, project deadlines)  
2. **Advanced Imputation:**  
   - Model-based or iterative imputation leveraging the strong fatigue–burnout correlation  
3. **Algorithm Exploration:**  
   - Experiment with LightGBM or CatBoost; ensemble stacking  
   - Bayesian hyperparameter optimization (e.g., Optuna)  
4. **Probability Calibration:**  
   - Use Platt scaling or isotonic regression to calibrate predicted class probabilities  
5. **Interpretability:**  
   - Integrate SHAP or LIME to explain individual predictions and feature impacts  
6. **Batch & API Services:**  
   - Add CSV upload for batch predictions; expose a RESTful JSON API (e.g., via FastAPI)  
7. **Containerization & Scalability:**  
   - Dockerize the app; deploy on Kubernetes or cloud services for higher concurrency  
8. **Security & Monitoring:**  
   - Implement authentication (SSO) and HTTPS; set up logging/alerts for data drift or unusual patterns  
9. **Longitudinal Forecasting:**  
   - If time-series data becomes available, predict future burnout risk using historical trends

---

**Conclusion:**  
The Employee Burnout Prediction project delivers a robust, well-evaluated XGBoost classifier integrated into both Flask and Streamlit applications—complete with preprocessing, EDA, and actionable recommendations. With targeted enhancements in data enrichment, model interpretability, and deployment infrastructure, this system can evolve into a comprehensive organizational tool for preventing burnout and fostering a healthier workforce.  
