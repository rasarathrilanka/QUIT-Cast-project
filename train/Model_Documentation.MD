# Machine Learning Models Documentation

## QuitCast: Employee Attrition Prediction System

### Overview
A multi-model ML system for predicting and analyzing employee attrition in Sri Lankan IT companies, trained on 100,000 realistic employee records with contextual economic and societal factors.

---

## 1. Main Attrition Prediction Model

### **Objective**
Binary classification to predict whether an employee will leave (attrition = 1) or stay (attrition = 0).

### **Algorithm**
- **Primary:** Random Forest Classifier
- **Comparison:** Gradient Boosting Classifier
- **Selection:** Model with higher AUC-ROC score selected automatically

### **Training Data**
- **Size:** 100,000 employees
- **Source:** Generated using Sri Lankan IT industry statistics (SLASSCOM 2022, LinkedIn Workforce 2022)
- **Split:** 80% training, 20% testing (stratified)
- **Features:** 15 features across 4 categories

### **Feature Engineering**

**Demographics (4 features):**
- `age`: 22-59 years
- `work_experience`: 0-20 years
- `time_at_current_role`: 0-15 years
- `marital_status_encoded`: Single/Married (Label Encoded)

**Job Characteristics (4 features):**
- `role_encoded`: 10 role types (Trainee to Architect)
- `department_encoded`: Engineering/QA/DevOps/Business
- `company_type_encoded`: Startup/Mid/Large/Multinational
- `wfh_available`: Binary (0/1)

**Satisfaction Factors (4 features):**
- `salary_satisfaction`: -1 to 1 scale (market gap)
- `career_growth_opportunity`: 0-10 scale
- `work_life_balance`: 0-10 scale
- `manager_relationship`: 0-10 scale

**External Factors (3 features):**
- `covid_impact_score`: 0-10 (based on joining year)
- `economic_crisis_impact`: 0-10 (2022 Sri Lanka crisis)
- `political_stability_concern`: 0-10 (timeline-based)

### **Hyperparameters**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    random_state=42,
    class_weight='balanced'
)
```

### **Target Variable Generation**
Ground truth labels created using weighted scoring system:
- Salary dissatisfaction: +30 points (if < -0.3)
- Career stagnation: +25 points (if < 4/10)
- Work-life imbalance: +15 points (if < 4/10)
- Manager issues: +10 points (if < 4/10)
- Economic crisis: +10 points (if > 7/10)
- Age < 28: +5 points
- Time in role > 4 years: +8 points
- 15% label noise added for realism
- **Threshold:** Score > 50 → attrition = 1

### **Model Validation**

**Cross-Validation:**
- 5-fold stratified CV
- Metric: AUC-ROC
- Expected CV Score: > 0.78

**Performance Metrics Tracked:**
- Accuracy: 75-80%
- AUC-ROC: > 0.80
- F1-Score: > 0.75
- Precision/Recall per class
- Confusion Matrix (all 4 quadrants)

**Visualizations Generated:**
- Feature Importance (bar chart)
- Confusion Matrix (heatmap)
- ROC Curve with AUC score
- Prediction Distribution (histogram)

### **Class Imbalance Handling**
- Attrition rate: 22-35% (realistic for IT industry)
- `class_weight='balanced'` to prevent majority class bias
- Stratified sampling in train/test split

---

## 2. Factor Contribution Model

### **Objective**
Multi-output regression to predict percentage contribution of each factor to an employee's attrition risk.

### **Algorithm**
- **Model:** Gradient Boosting Regressor wrapped in MultiOutputRegressor
- **Reason:** GBR doesn't natively support multiple outputs; wrapper enables parallel prediction of 7 targets

### **Training Data**
- **Size:** 10,000 samples
- **Source:** Research-based weights from HR literature
- **Base Contributions:**
  - Career Development: 22% (LinkedIn 2022, Work Institute 2020)
  - Salary: 18% (McKinsey Great Attrition 2022)
  - Work-Life Balance: 15%
  - Economic Crisis: 15% (Sri Lanka specific)
  - Manager Relationship: 12%
  - Political Instability: 10%
  - COVID Impact: 8%

### **Feature Set** (11 inputs)
- `age`, `work_experience`, `time_in_role`
- `salary_satisfaction`, `career_growth_opportunity`, `work_life_balance`, `manager_relationship`
- `covid_impact_score`, `economic_crisis_impact`, `political_stability_concern`
- `wfh_available`

### **Target Variables** (7 outputs)
Each represents percentage contribution (sum = 100%):
- `salary_contribution`
- `career_contribution`
- `work_life_contribution`
- `manager_contribution`
- `covid_contribution`
- `economic_contribution`
- `political_contribution`

### **Hyperparameters**
```python
MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    n_jobs=-1
)
```

### **Dynamic Adjustment Logic**
Base contributions adjusted by:
- **Severity scaling:** High dissatisfaction increases contribution
- **Age-based:** Young employees prioritize salary/career; seniors prioritize stability
- **WFH effect:** Reduces work-life issues by 30%
- **Normalization:** All outputs sum to 100%

### **Validation Metrics**
- **MAE (Mean Absolute Error):** 3-5% per factor
- **R² Score:** > 0.85
- **Feature Importance:** Tracked to understand prediction drivers

**Visualizations:**
- Feature Importance by factor
- Actual vs Predicted scatter plots per target
- Residual distribution

---

## 3. Retention Strategy Recommender Model

### **Objective**
Multi-class classification to recommend the most effective retention strategy for a given employee profile.

### **Algorithm**
- **Model:** Random Forest Classifier
- **Output:** Probability distribution over 10 strategy types
- **Top-3 Selection:** Returns 3 highest probability strategies with confidence scores

### **Training Data**
- **Size:** 5,000 samples
- **Source:** HR intervention studies (SHRM 2021, Gallup 2022, HBR research)
- **Strategy Effectiveness Rates:** Evidence-based from literature

### **10 Strategy Classes**

| Strategy | Effectiveness | Cost | Best For |
|----------|---------------|------|----------|
| Salary Increase | 75% | $10-25K | salary_satisfaction < 4 |
| Promotion | 70% | $15-30K | career_growth < 4, time_in_role > 3 |
| WFH Flexibility | 65% | $2-5K | work_life_balance < 5, no WFH |
| Manager Change | 60% | $5-10K | manager_relationship < 4 |
| Career Development | 68% | $8-15K | career_growth < 5, age < 40 |
| Wellness Program | 55% | $5-12K | economic/covid impact > 6 |
| Recognition/Reward | 58% | $3-8K | manager_relationship < 6 |
| Team Transfer | 62% | $4-9K | manager/WLB issues |
| Retention Bonus | 50% | $8-20K | critical_skill, salary < 5 |
| Equity Options | 72% | $20-50K | career + salary issues |

### **Feature Set** (14 inputs)
- Demographics: `age`, `work_experience`, `time_in_role`, `tenure`
- Satisfaction: `salary_satisfaction`, `career_growth`, `work_life_balance`, `manager_relationship`
- External: `covid_impact`, `economic_impact`, `political_concern`
- Job: `wfh_available`, `critical_skill`
- Risk: `attrition_risk` (from main model)

### **Strategy Selection Logic**
1. Identify dominant issue (salary/career/manager/work-life/external)
2. Map issue to top 3 strategies based on research
3. Calculate confidence = issue clarity score (50-95%)
4. Return ranked by confidence

### **Hyperparameters**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
```

### **Validation Metrics**
- **Accuracy:** 75-85%
- **Average Confidence:** ~70%
- **Confusion Matrix:** 10x10 for all strategy pairs
- **Feature Importance:** Which factors drive recommendations

**Visualizations:**
- Strategy Confusion Matrix
- Feature Importance ranking
- Confidence distribution histogram

---

## Model Integration & Deployment

### **Backward Compatibility**
All models support graceful degradation:
- Main model: Auto-fills missing 9 features if only 6 provided
- Factor model: Returns research-based defaults if unavailable
- Strategy model: Returns top-3 generic strategies as fallback

### **Production Pipeline**
1. **Input:** 6 core features (frontend form)
2. **Auto-fill:** `fill_missing_features()` infers 9 contextual factors
3. **Main Model:** Predicts attrition probability
4. **Factor Model:** Explains contribution breakdown
5. **Strategy Model:** Recommends top-3 interventions
6. **Output:** Unified JSON with all insights

### **Custom Model Training**
Users can train custom models with their data:
- **Minimum samples:** 100 (validated)
- **Minimum per class:** 10 samples each
- **Accuracy threshold:** Warning if < 65%
- **Model persistence:** Saved with metadata (accuracy, AUC, timestamp)
- **Model management:** List, load, delete custom models

### **Quality Assurance**

**Data Validation:**
- Column presence checks
- Data type validation
- Range checks (e.g., age 18-70)
- Missing value handling

**Model Monitoring:**
- Accuracy tracking over time
- Prediction distribution monitoring
- Feature drift detection (planned)

**Research Foundation:**
- SLASSCOM HR Survey 2022
- LinkedIn Workforce Sri Lanka 2022
- Work Institute Retention Report 2020
- McKinsey Great Attrition Research 2022
- SHRM Employee Retention Survey 2021
- Gallup State of the Workplace 2022
- World Bank Sri Lanka Economic Crisis Report 2022

---

## Technical Stack

**Backend:**
- scikit-learn 1.3+
- pandas, numpy
- Flask REST API
- joblib for model serialization

**Frontend:**
- React 18
- Recharts for visualizations
- Real-time API integration

**Model Artifacts:**
- `.pkl` files with model + encoders + metadata
- Metadata includes: feature names, training metrics, timestamp
- Average model size: 5-20 MB

---

## Future Enhancements
- SHAP values for individual prediction explanations
- A/B testing framework for strategy effectiveness
- Automated retraining pipeline
- Real-time model performance dashboards
- Ensemble methods combining multiple custom models

---

**Publication Ready:** All models trained on research-backed data with proper citations, reproducible methodology, and comprehensive evaluation metrics.