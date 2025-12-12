"""
Flask Backend API for Employee Attrition Prediction System
Provides REST API endpoints for predictions and analytics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained model
MODEL_PATH = '../models/attrition_model.pkl'
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
label_encoders = model_data['label_encoders']
feature_names = model_data['feature_names']
# Store last uploaded data in memory (use Redis in production)
last_uploaded_data = None
print("✓ Model loaded successfully")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ==================== HELPER FUNCTIONS ====================

def get_risk_level(probability):
    """Convert probability to risk level with color"""
    if probability < 15:
        return {"level": "Low Risk", "color": "#10b981", "icon": "🟢"}
    elif probability < 30:
        return {"level": "Medium Risk", "color": "#f59e0b", "icon": "🟡"}
    elif probability < 50:
        return {"level": "High Risk", "color": "#f97316", "icon": "🟠"}
    else:
        return {"level": "Very High Risk", "color": "#ef4444", "icon": "🔴"}


def predict_single(employee_data):
    """Predict attrition for a single employee"""
    # Create DataFrame
    df = pd.DataFrame([employee_data])

    # Encode categorical variables
    df['marital_status_encoded'] = label_encoders['marital_status'].transform(df['marital_status'])
    df['role_encoded'] = label_encoders['role'].transform(df['role'])

    # Select features
    X = df[feature_names]

    # Predict
    attrition_prob = model.predict_proba(X)[0, 1] * 100
    prediction = model.predict(X)[0]

    # Get risk level
    risk = get_risk_level(attrition_prob)

    return {
        'attrition_probability': round(attrition_prob, 2),
        'will_leave': bool(prediction),
        'risk_level': risk['level'],
        'risk_color': risk['color'],
        'risk_icon': risk['icon'],
        'confidence': 'High' if abs(attrition_prob - 50) > 30 else 'Medium'
    }


# Update feature_names to include new factors
feature_names_extended = feature_names + [
    'salary_satisfaction',
    'covid_impact_score',
    'economic_crisis_impact',
    'political_stability_concern'
]


# ==================== RETENTION OPTIMIZER ====================

def generate_retention_strategies(employee_data):
    """
    Generate personalized retention strategies using counterfactual analysis
    """
    strategies = []
    current_prob = calculate_attrition_with_factors(employee_data)['probability']

    # Strategy 1: Salary Increase
    salary_scenarios = []
    for increase in [0.10, 0.15, 0.20, 0.25]:
        new_data = employee_data.copy()
        current_sal = new_data.get('salary_satisfaction', -0.2)
        new_data['salary_satisfaction'] = min(current_sal + increase * 2, 1.0)  # Cap at 1.0
        new_prob = calculate_attrition_with_factors(new_data)['probability']

        salary_scenarios.append({
            'increase_percentage': int(increase * 100),
            'new_risk': round(new_prob, 1),
            'reduction': round(current_prob - new_prob, 1),
            'cost': int(increase * 100000),  # Assume 100K base salary
            'roi': 0
        })

    # Calculate ROI (replacement cost = 150% of salary)
    replacement_cost = 150000
    for scenario in salary_scenarios:
        potential_savings = replacement_cost if scenario['reduction'] > 20 else replacement_cost * (
                    scenario['reduction'] / 50)
        scenario['roi'] = int((potential_savings - scenario['cost']) / scenario['cost'] * 100) if scenario[
                                                                                                      'cost'] > 0 else 0

    # Select best salary strategy
    best_salary = max(salary_scenarios, key=lambda x: x['roi'])
    strategies.append({
        'type': 'salary',
        'name': f"Salary Increase (+{best_salary['increase_percentage']}%)",
        'icon': '💰',
        'current_risk': round(current_prob, 1),
        'new_risk': best_salary['new_risk'],
        'reduction': best_salary['reduction'],
        'cost': best_salary['cost'],
        'savings': 150000 if best_salary['reduction'] > 20 else int(150000 * (best_salary['reduction'] / 50)),
        'roi': best_salary['roi'],
        'priority': 1 if best_salary['reduction'] > 15 else 2,
        'description': 'Address salary dissatisfaction through competitive compensation'
    })

    # Strategy 2: Enable WFH
    if employee_data.get('wfh_available', 1) == 0:
        new_data = employee_data.copy()
        new_data['wfh_available'] = 1
        new_prob = calculate_attrition_with_factors(new_data)['probability']
        reduction = current_prob - new_prob

        strategies.append({
            'type': 'wfh',
            'name': 'Enable Work From Home',
            'icon': '🏠',
            'current_risk': round(current_prob, 1),
            'new_risk': round(new_prob, 1),
            'reduction': round(reduction, 1),
            'cost': 2000,  # Setup cost
            'savings': 150000 if reduction > 15 else int(150000 * (reduction / 50)),
            'roi': int((150000 * (reduction / 50) - 2000) / 2000 * 100) if reduction > 5 else 0,
            'priority': 1 if reduction > 10 else 3,
            'description': 'Improve work-life balance through remote work flexibility'
        })

    # Strategy 3: Role Change/Promotion
    new_data = employee_data.copy()
    current_role = new_data.get('role', 'Software Engineer')

    # Promotion mapping
    promotions = {
        'Junior Developer': 'Software Engineer',
        'Software Engineer': 'Senior Software Engineer',
        'Senior Software Engineer': 'Tech Lead',
        'QA Engineer': 'Senior QA Engineer',
        'Senior QA Engineer': 'QA Lead',
        'Tech Lead': 'Engineering Manager',
        'Trainee Developer': 'Junior Developer'
    }

    if current_role in promotions:
        new_data['role'] = promotions[current_role]
        new_data['time_at_current_role'] = 0.1  # Fresh in new role
        new_prob = calculate_attrition_with_factors(new_data)['probability']
        reduction = current_prob - new_prob

        strategies.append({
            'type': 'promotion',
            'name': f"Promote to {promotions[current_role]}",
            'icon': '📈',
            'current_risk': round(current_prob, 1),
            'new_risk': round(new_prob, 1),
            'reduction': round(reduction, 1),
            'cost': 15000,  # Promotion cost (salary bump + admin)
            'savings': 150000 if reduction > 15 else int(150000 * (reduction / 50)),
            'roi': int((150000 * (reduction / 50) - 15000) / 15000 * 100) if reduction > 5 else 0,
            'priority': 1 if reduction > 12 else 2,
            'description': 'Career advancement to reduce stagnation and increase engagement'
        })

    # Strategy 4: Reduce Economic/External Stress
    new_data = employee_data.copy()
    new_data['economic_crisis_impact'] = max(new_data.get('economic_crisis_impact', 5) - 3, 0)
    new_data['political_stability_concern'] = max(new_data.get('political_stability_concern', 5) - 2, 0)
    new_prob = calculate_attrition_with_factors(new_data)['probability']
    reduction = current_prob - new_prob

    strategies.append({
        'type': 'support',
        'name': 'Financial Wellness Program',
        'icon': '🛡️',
        'current_risk': round(current_prob, 1),
        'new_risk': round(new_prob, 1),
        'reduction': round(reduction, 1),
        'cost': 5000,  # Wellness program cost
        'savings': 150000 if reduction > 10 else int(150000 * (reduction / 50)),
        'roi': int((150000 * (reduction / 50) - 5000) / 5000 * 100) if reduction > 3 else 0,
        'priority': 2,
        'description': 'Support employees through economic uncertainty with financial planning and benefits'
    })

    # Sort by priority and ROI
    strategies.sort(key=lambda x: (-x['priority'], -x['roi']))

    return {
        'employee_data': employee_data,
        'current_risk': round(current_prob, 1),
        'strategies': strategies[:4],  # Top 4 strategies
        'combined_impact': {
            'implementing_top_3': round(current_prob - sum(s['reduction'] for s in strategies[:3]) * 0.7, 1),
            'total_cost': sum(s['cost'] for s in strategies[:3]),
            'total_savings': sum(s['savings'] for s in strategies[:3]),
            'net_benefit': sum(s['savings'] for s in strategies[:3]) - sum(s['cost'] for s in strategies[:3])
        }
    }


# ==================== CLUSTERING & SEGMENTATION ====================

def perform_employee_clustering(employees_data):
    """
    Perform K-Means clustering to identify employee risk profiles
    """
    if len(employees_data) < 10:
        return {'error': 'Need at least 10 employees for clustering'}

    df = pd.DataFrame(employees_data)

    # Make predictions first
    df['marital_status_encoded'] = label_encoders['marital_status'].transform(df['marital_status'])
    df['role_encoded'] = label_encoders['role'].transform(df['role'])
    X_pred = df[feature_names]
    df['attrition_probability'] = model.predict_proba(X_pred)[:, 1] * 100

    # Features for clustering
    cluster_features = [
        'age', 'work_experience', 'time_at_current_role',
        'salary_satisfaction', 'covid_impact_score',
        'economic_crisis_impact', 'political_stability_concern',
        'attrition_probability'
    ]

    X_cluster = df[cluster_features].fillna(df[cluster_features].mean())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Determine optimal number of clusters (3-5)
    optimal_k = min(5, max(3, len(df) // 20))

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Analyze each cluster
    clusters = []
    for cluster_id in range(optimal_k):
        cluster_df = df[df['cluster'] == cluster_id]

        # Identify dominant characteristics
        avg_age = cluster_df['age'].mean()
        avg_exp = cluster_df['work_experience'].mean()
        avg_risk = cluster_df['attrition_probability'].mean()
        avg_salary_sat = cluster_df['salary_satisfaction'].mean()
        avg_covid = cluster_df['covid_impact_score'].mean()
        avg_economic = cluster_df['economic_crisis_impact'].mean()

        # Generate cluster name based on characteristics
        cluster_name = generate_cluster_name(avg_age, avg_exp, avg_risk, avg_salary_sat, avg_covid, avg_economic)

        # Top factors driving this cluster
        factors = []
        if avg_salary_sat < -0.2:
            factors.append('Salary Dissatisfaction')
        if avg_covid > 6:
            factors.append('High COVID Impact')
        if avg_economic > 6:
            factors.append('Economic Crisis Affected')
        if avg_age < 28:
            factors.append('Junior Level')
        if avg_exp > 8:
            factors.append('Senior Level')
        if cluster_df['time_at_current_role'].mean() > 4:
            factors.append('Role Stagnation')

        # Recommended actions
        actions = generate_cluster_actions(avg_salary_sat, avg_covid, avg_economic, avg_risk)

        clusters.append({
            'id': int(cluster_id),
            'name': cluster_name,
            'size': len(cluster_df),
            'avg_risk': round(avg_risk, 1),
            'risk_level': 'Critical' if avg_risk > 60 else 'High' if avg_risk > 40 else 'Medium' if avg_risk > 25 else 'Low',
            'characteristics': {
                'avg_age': round(avg_age, 1),
                'avg_experience': round(avg_exp, 1),
                'avg_salary_satisfaction': round(avg_salary_sat, 2),
                'avg_covid_impact': round(avg_covid, 1),
                'avg_economic_impact': round(avg_economic, 1)
            },
            'key_factors': factors,
            'recommended_actions': actions,
            'top_employees': cluster_df.nlargest(5, 'attrition_probability')[
                ['employee_id', 'role', 'attrition_probability']
            ].to_dict('records')
        })

    # Sort clusters by risk
    clusters.sort(key=lambda x: -x['avg_risk'])

    # Anomaly detection for early warning
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(X_scaled)
    anomalies = df[df['anomaly'] == -1]

    return {
        'clusters': clusters,
        'total_employees': len(df),
        'cluster_distribution': df['cluster'].value_counts().to_dict(),
        'early_warnings': {
            'anomalies_detected': len(anomalies),
            'sudden_risk_employees': anomalies.nlargest(5, 'attrition_probability')[
                ['employee_id', 'role', 'attrition_probability']
            ].to_dict('records') if len(anomalies) > 0 else []
        }
    }


def generate_cluster_name(avg_age, avg_exp, avg_risk, avg_salary_sat, avg_covid, avg_economic):
    """Generate descriptive cluster name"""
    if avg_risk > 60:
        risk_term = "Flight Risk"
    elif avg_risk > 40:
        risk_term = "High Risk"
    elif avg_risk > 25:
        risk_term = "Moderate Risk"
    else:
        risk_term = "Stable"

    if avg_age < 28 and avg_exp < 3:
        group = "Juniors"
    elif avg_age < 35 and avg_exp < 7:
        group = "Mid-Level"
    else:
        group = "Seniors"

    # Add dominant factor
    if avg_salary_sat < -0.3:
        return f"{risk_term} {group} (Underpaid)"
    elif avg_covid > 7:
        return f"{risk_term} {group} (COVID Affected)"
    elif avg_economic > 7:
        return f"{risk_term} {group} (Economic Crisis)"
    else:
        return f"{risk_term} {group}"


def generate_cluster_actions(avg_salary_sat, avg_covid, avg_economic, avg_risk):
    """Generate recommended actions for cluster"""
    actions = []

    if avg_salary_sat < -0.2:
        actions.append("Conduct immediate salary review and market adjustment")

    if avg_covid > 6:
        actions.append("Provide mental health support and flexible work arrangements")

    if avg_economic > 6:
        actions.append("Implement financial wellness programs and cost-of-living adjustments")

    if avg_risk > 50:
        actions.append("Schedule one-on-one retention conversations with manager")
        actions.append("Fast-track promotion opportunities where applicable")

    if len(actions) == 0:
        actions.append("Continue monitoring - maintain current engagement initiatives")

    return actions

def calculate_attrition_with_factors(employee_data):
    """
    Calculate attrition with breakdown of contributing factors
    Returns: probability and factor contributions
    """
    # Base prediction from original model
    base_prob = 0.15

    # Factor contributions (each 0-1 scale)
    factors = {}

    # 1. Salary Satisfaction Impact (strongest factor)
    salary_sat = employee_data.get('salary_satisfaction', 0)
    if salary_sat < -0.3:  # Significantly underpaid
        salary_impact = 0.25 * abs(salary_sat)
    elif salary_sat < 0:  # Slightly underpaid
        salary_impact = 0.15 * abs(salary_sat)
    else:  # Paid well
        salary_impact = -0.05 * salary_sat
    factors['salary'] = max(0, min(salary_impact, 0.3))

    # 2. COVID Impact
    covid_score = employee_data.get('covid_impact_score', 0) / 10
    covid_impact = covid_score * 0.15  # Max 15% contribution
    factors['covid'] = covid_impact

    # 3. Economic Crisis Impact
    economic_score = employee_data.get('economic_crisis_impact', 0) / 10
    economic_impact = economic_score * 0.20  # Max 20% contribution
    factors['economic'] = economic_impact

    # 4. Political Stability
    political_score = employee_data.get('political_stability_concern', 0) / 10
    political_impact = political_score * 0.10  # Max 10% contribution
    factors['political'] = political_impact

    # Other job-related factors from original model
    job_factors = 0

    # Work-life balance
    if employee_data.get('wfh_available', 1) == 0:
        job_factors += 0.08

    # Time in role (stagnation)
    if employee_data.get('time_at_current_role', 0) > 4:
        job_factors += 0.10

    # Age factor
    age = employee_data.get('age', 30)
    if age < 26:
        job_factors += 0.08

    # Marital status (mobility)
    if employee_data.get('marital_status', '') == 'Single':
        job_factors += 0.06

    factors['job_factors'] = job_factors

    # Total probability
    total_prob = base_prob + sum(factors.values())
    total_prob = np.clip(total_prob, 0.01, 0.95)

    # Normalize factor contributions to percentages
    total_contribution = sum(factors.values())
    if total_contribution > 0:
        factor_percentages = {
            k: round((v / total_contribution) * 100, 1)
            for k, v in factors.items()
        }
    else:
        factor_percentages = {k: 0 for k in factors.keys()}

    return {
        'probability': round(total_prob * 100, 2),
        'factors': factor_percentages,
        'raw_factors': factors
    }
def calculate_future_attrition(current_employees, quarters_ahead=4):
    """
    Calculate predicted attrition for future quarters
    Simulates how attrition might change over time
    """
    predictions = []

    for quarter in range(1, quarters_ahead + 1):
        # Simulate aging and experience increase
        future_employees = current_employees.copy()
        years_forward = quarter * 0.25  # Each quarter = 0.25 years

        future_employees['age'] = future_employees['age'] + (years_forward * 1)
        future_employees['work_experience'] = future_employees['work_experience'] + years_forward
        future_employees['time_at_current_role'] = future_employees['time_at_current_role'] + years_forward

        # Encode and predict
        future_employees['marital_status_encoded'] = label_encoders['marital_status'].transform(
            future_employees['marital_status']
        )
        future_employees['role_encoded'] = label_encoders['role'].transform(future_employees['role'])

        X = future_employees[feature_names]
        probs = model.predict_proba(X)[:, 1] * 100

        # Calculate quarter metrics
        avg_prob = probs.mean()
        expected_leavers = (probs > 50).sum()

        # Generate quarter label
        current_date = datetime.now()
        future_date = current_date + timedelta(days=90 * quarter)
        quarter_label = f"Q{((future_date.month - 1) // 3) + 1} {future_date.year}"

        predictions.append({
            'quarter': quarter_label,
            'quarter_number': quarter,
            'avg_attrition_probability': round(avg_prob, 2),
            'expected_leavers': int(expected_leavers),
            'total_employees': len(future_employees)
        })

    return predictions


# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict/single', methods=['POST'])
def predict_single_employee():
    """
    Predict attrition for a single employee

    Request body:
    {
        "age": 28,
        "time_at_current_role": 1.5,
        "marital_status": "Single",
        "role": "Software Engineer",
        "work_experience": 3.0,
        "wfh_available": 1,
        "employee_name": "John Doe" (optional)
    }
    """
    try:
        data = request.json

        # Validate required fields
        required_fields = ['age', 'time_at_current_role', 'marital_status',
                           'role', 'work_experience', 'wfh_available']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Make prediction
        result = predict_single(data)

        # Add employee name if provided
        if 'employee_name' in data:
            result['employee_name'] = data['employee_name']

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def predict_batch_employees():
    """
    Predict attrition for multiple employees

    Request body:
    {
        "employees": [
            {
                "age": 28,
                "time_at_current_role": 1.5,
                "marital_status": "Single",
                "role": "Software Engineer",
                "work_experience": 3.0,
                "wfh_available": 1,
                "employee_name": "John Doe"
            },
            ...
        ]
    }
    """
    try:
        data = request.json

        if 'employees' not in data:
            return jsonify({'error': 'Missing employees array'}), 400

        employees = data['employees']
        results = []

        for emp in employees:
            prediction = predict_single(emp)
            prediction['employee_data'] = emp
            results.append(prediction)

        # Calculate aggregate statistics
        probabilities = [r['attrition_probability'] for r in results]

        summary = {
            'total_employees': len(results),
            'avg_attrition_probability': round(np.mean(probabilities), 2),
            'expected_leavers': sum(1 for r in results if r['will_leave']),
            'high_risk_count': sum(1 for r in results if r['attrition_probability'] >= 50),
            'medium_risk_count': sum(1 for r in results if 30 <= r['attrition_probability'] < 50),
            'low_risk_count': sum(1 for r in results if r['attrition_probability'] < 30)
        }

        return jsonify({
            'predictions': results,
            'summary': summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/department', methods=['POST'])
def predict_department():
    """
    Predict attrition for a department/team
    Includes future forecasting

    Request body:
    {
        "department_name": "Engineering",
        "employees": [...],
        "forecast_quarters": 4 (optional, default 4)
    }
    """
    try:
        data = request.json

        if 'employees' not in data:
            return jsonify({'error': 'Missing employees array'}), 400

        employees = data['employees']
        department_name = data.get('department_name', 'Department')
        forecast_quarters = data.get('forecast_quarters', 4)

        # Current predictions
        df = pd.DataFrame(employees)

        # Make predictions
        df['marital_status_encoded'] = label_encoders['marital_status'].transform(df['marital_status'])
        df['role_encoded'] = label_encoders['role'].transform(df['role'])
        X = df[feature_names]

        probabilities = model.predict_proba(X)[:, 1] * 100
        predictions = model.predict(X)

        # Add predictions to dataframe
        df['attrition_probability'] = probabilities
        df['predicted_attrition'] = predictions
        df['risk_level'] = df['attrition_probability'].apply(
            lambda x: get_risk_level(x)['level']
        )

        # Current summary
        current_summary = {
            'department_name': department_name,
            'total_employees': len(df),
            'avg_attrition_probability': round(probabilities.mean(), 2),
            'expected_leavers': int(predictions.sum()),
            'high_risk_count': int((probabilities >= 50).sum()),
            'medium_risk_count': int(((probabilities >= 30) & (probabilities < 50)).sum()),
            'low_risk_count': int((probabilities < 30).sum())
        }

        # Future forecasting
        future_predictions = calculate_future_attrition(df, forecast_quarters)

        # Top risk employees
        top_risk = df.nlargest(10, 'attrition_probability')[
            ['age', 'role', 'work_experience', 'attrition_probability', 'risk_level']
        ].to_dict('records')

        return jsonify({
            'current': current_summary,
            'future_forecast': future_predictions,
            'top_risk_employees': top_risk,
            'all_predictions': df[['age', 'role', 'work_experience', 'time_at_current_role',
                                   'attrition_probability', 'risk_level']].to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload/csv', methods=['POST'])
def upload_csv():
    """Upload CSV and store for analytics"""
    global last_uploaded_data

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400

        # Read CSV
        df = pd.read_csv(file)

        # Validate required columns
        required_cols = ['employee_id', 'age', 'time_at_current_role', 'marital_status',
                         'role', 'work_experience', 'wfh_available']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return jsonify({'error': f'Missing columns: {", ".join(missing_cols)}'}), 400

        # Store data for later use
        last_uploaded_data = df.to_dict('records')

        # Generate predictions for 5 years (20 quarters)
        results = []

        for year in range(1, 6):
            year_label = datetime.now().year + year

            for quarter in range(1, 5):
                future_df = df.copy()
                years_forward = (year - 1) + (quarter * 0.25)

                future_df['age'] = future_df['age'] + years_forward
                future_df['work_experience'] = future_df['work_experience'] + years_forward
                future_df['time_at_current_role'] = future_df['time_at_current_role'] + years_forward

                future_df['marital_status_encoded'] = label_encoders['marital_status'].transform(
                    future_df['marital_status']
                )
                future_df['role_encoded'] = label_encoders['role'].transform(future_df['role'])

                X = future_df[feature_names]
                probs = model.predict_proba(X)[:, 1] * 100

                avg_prob = probs.mean()
                expected_leavers = (probs > 50).sum()

                # Inside the quarter loop, update the employees list:
                employees = []
                for idx, row in future_df.iterrows():
                    employees.append({
                        'employee_id': row['employee_id'],
                        'role': row['role'],
                        'age': int(row['age']),
                        'work_experience': round(float(row['work_experience']), 2),
                        'attrition_probability': round(probs[idx], 2)
                    })

                results.append({
                    'year': year_label,
                    'quarter': quarter,
                    'quarter_label': f'Q{quarter} {year_label}',
                    'avg_attrition_probability': round(avg_prob, 2),
                    'expected_leavers': int(expected_leavers),
                    'total_employees': len(future_df),
                    'employees': employees
                })

        years_summary = {}
        for r in results:
            year = r['year']
            if year not in years_summary:
                years_summary[year] = {
                    'year': year,
                    'quarters': [],
                    'avg_attrition_probability': 0
                }
            years_summary[year]['quarters'].append({
                'quarter': r['quarter'],
                'quarter_label': r['quarter_label'],
                'avg_attrition_probability': r['avg_attrition_probability'],
                'expected_leavers': r['expected_leavers']
            })

        for year, data in years_summary.items():
            data['avg_attrition_probability'] = round(
                sum(q['avg_attrition_probability'] for q in data['quarters']) / len(data['quarters']),
                2
            )

        return jsonify({
            'total_employees': len(df),
            'years': list(years_summary.values()),
            'all_predictions': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics/uploaded', methods=['GET'])
def get_uploaded_analytics():
    """Get analytics from last uploaded data"""
    global last_uploaded_data

    if last_uploaded_data is None:
        return jsonify({'error': 'No data uploaded yet'}), 404

    try:
        df = pd.DataFrame(last_uploaded_data)

        # Make predictions
        df['marital_status_encoded'] = label_encoders['marital_status'].transform(df['marital_status'])
        df['role_encoded'] = label_encoders['role'].transform(df['role'])
        X = df[feature_names]

        probabilities = model.predict_proba(X)[:, 1] * 100
        df['attrition_probability'] = probabilities

        # Get ALL employees with predictions for modal display
        all_employees = df[['employee_id', 'age', 'role', 'work_experience', 'attrition_probability']].to_dict(
            'records')

        # High risk (>= 50%)
        high_risk = [emp for emp in all_employees if emp['attrition_probability'] >= 50]

        # Medium risk (30-50%)
        medium_risk = [emp for emp in all_employees if 30 <= emp['attrition_probability'] < 50]

        # Overall stats
        analytics = {
            'total_employees': len(df),
            'overall_attrition_rate': round(probabilities.mean(), 2),
            'high_risk_count': len(high_risk),
            'medium_risk_count': len(medium_risk),
            'low_risk_count': int((probabilities < 30).sum()),

            # Full lists for modal
            'high_risk_employees': sorted(high_risk, key=lambda x: x['attrition_probability'], reverse=True),
            'medium_risk_employees': sorted(medium_risk, key=lambda x: x['attrition_probability'], reverse=True),

            # By role
            'by_role': df.groupby('role')['attrition_probability'].agg(['mean', 'count']).round(2).to_dict('index'),

            # By department (infer from role)
            'by_department': {},

            # Top risk employees (for existing display)
            'top_risk': df.nlargest(10, 'attrition_probability')[
                ['employee_id', 'age', 'role', 'work_experience', 'attrition_probability']
            ].to_dict('records')
        }

        # Categorize into departments
        engineering_roles = ['Junior Developer', 'Software Engineer', 'Senior Software Engineer',
                             'Tech Lead', 'Engineering Manager', 'DevOps Engineer', 'Architect', 'Trainee Developer']
        qa_roles = ['QA Engineer', 'Senior QA Engineer']
        business_roles = ['Business Analyst', 'Product Manager', 'Director']

        df['department'] = df['role'].apply(lambda x:
                                            'Engineering' if x in engineering_roles else
                                            'QA' if x in qa_roles else
                                            'Business' if x in business_roles else 'Other'
                                            )

        analytics['by_department'] = df.groupby('department')['attrition_probability'].agg(['mean', 'count']).round(
            2).to_dict('index')

        return jsonify(analytics)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/employee/<employee_id>', methods=['GET'])
def predict_by_employee_id(employee_id):
    """Get prediction for specific employee by ID"""
    global last_uploaded_data

    if last_uploaded_data is None:
        return jsonify({'error': 'No data uploaded yet'}), 404

    try:
        df = pd.DataFrame(last_uploaded_data)
        employee = df[df['employee_id'] == employee_id]

        if employee.empty:
            return jsonify({'error': f'Employee {employee_id} not found'}), 404

        employee_data = employee.iloc[0].to_dict()
        result = predict_single(employee_data)
        result['employee_data'] = employee_data

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/department/<department_name>', methods=['GET'])
def predict_by_department(department_name):
    """Get 5-year predictions by department"""
    global last_uploaded_data

    if last_uploaded_data is None:
        return jsonify({'error': 'No data uploaded yet'}), 404

    try:
        df = pd.DataFrame(last_uploaded_data)

        # Categorize departments
        engineering_roles = ['Junior Developer', 'Software Engineer', 'Senior Software Engineer',
                             'Tech Lead', 'Engineering Manager', 'DevOps Engineer', 'Architect', 'Trainee Developer']
        qa_roles = ['QA Engineer', 'Senior QA Engineer']
        business_roles = ['Business Analyst', 'Product Manager', 'Director']

        df['department'] = df['role'].apply(lambda x:
                                            'Engineering' if x in engineering_roles else
                                            'QA' if x in qa_roles else
                                            'Business' if x in business_roles else 'Other'
                                            )

        # Filter by department
        dept_df = df[df['department'] == department_name]

        if dept_df.empty:
            return jsonify({'error': f'No employees in {department_name} department'}), 404

        # Generate 5-year predictions
        results = []

        for year in range(1, 6):
            year_label = datetime.now().year + year

            for quarter in range(1, 5):
                future_df = dept_df.copy()
                years_forward = (year - 1) + (quarter * 0.25)

                future_df['age'] = future_df['age'] + years_forward
                future_df['work_experience'] = future_df['work_experience'] + years_forward
                future_df['time_at_current_role'] = future_df['time_at_current_role'] + years_forward

                future_df['marital_status_encoded'] = label_encoders['marital_status'].transform(
                    future_df['marital_status']
                )
                future_df['role_encoded'] = label_encoders['role'].transform(future_df['role'])

                X = future_df[feature_names]
                probs = model.predict_proba(X)[:, 1] * 100

                avg_prob = probs.mean()
                expected_leavers = (probs > 50).sum()

                results.append({
                    'year': year_label,
                    'quarter': quarter,
                    'quarter_label': f'Q{quarter} {year_label}',
                    'avg_attrition_probability': round(avg_prob, 2),
                    'expected_leavers': int(expected_leavers),
                    'total_employees': len(dept_df)
                })

        # Group by year
        years_summary = {}
        for r in results:
            year = r['year']
            if year not in years_summary:
                years_summary[year] = {
                    'year': year,
                    'quarters': [],
                    'avg_attrition_probability': 0
                }
            years_summary[year]['quarters'].append({
                'quarter': r['quarter'],
                'quarter_label': r['quarter_label'],
                'avg_attrition_probability': r['avg_attrition_probability'],
                'expected_leavers': r['expected_leavers']
            })

        for year, data in years_summary.items():
            data['avg_attrition_probability'] = round(
                sum(q['avg_attrition_probability'] for q in data['quarters']) / len(data['quarters']),
                2
            )

        return jsonify({
            'department': department_name,
            'total_employees': len(dept_df),
            'years': list(years_summary.values())
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/sample-csv', methods=['GET'])
def download_sample_csv():
    """Download sample CSV template"""
    sample_data = {
        'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
        'age': [28, 35, 24, 40, 32],
        'time_at_current_role': [1.5, 3.0, 0.5, 2.0, 4.5],
        'marital_status': ['Single', 'Married', 'Single', 'Married', 'Single'],
        'role': ['Software Engineer', 'Senior Software Engineer', 'Junior Developer', 'Tech Lead', 'QA Engineer'],
        'work_experience': [3.0, 7.0, 1.2, 10.0, 5.5],
        'wfh_available': [1, 1, 0, 1, 1]
    }

    df = pd.DataFrame(sample_data)

    # Convert to CSV
    csv = df.to_csv(index=False)

    from flask import Response
    return Response(
        csv,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sample_employees.csv'}
    )

@app.route('/api/analytics/overview', methods=['POST'])
def get_analytics_overview():
    """
    Get overall analytics and insights

    Request body:
    {
        "employees": [...]
    }
    """
    try:
        data = request.json

        if 'employees' not in data:
            return jsonify({'error': 'Missing employees array'}), 400

        df = pd.DataFrame(data['employees'])

        # Make predictions
        df['marital_status_encoded'] = label_encoders['marital_status'].transform(df['marital_status'])
        df['role_encoded'] = label_encoders['role'].transform(df['role'])
        X = df[feature_names]

        probabilities = model.predict_proba(X)[:, 1] * 100
        df['attrition_probability'] = probabilities

        # Analytics
        analytics = {
            'total_employees': len(df),
            'overall_attrition_rate': round(probabilities.mean(), 2),
            'total_predicted_leavers': int((probabilities > 50).sum()),

            # By role
            'by_role': df.groupby('role')['attrition_probability'].agg(['mean', 'count']).round(2).to_dict('index'),

            # By marital status
            'by_marital_status': df.groupby('marital_status')['attrition_probability'].agg(['mean', 'count']).round(
                2).to_dict('index'),

            # By age group
            'by_age_group': df.groupby(pd.cut(df['age'], bins=[20, 25, 30, 35, 40, 50, 65]))[
                'attrition_probability'].agg(['mean', 'count']).round(2).to_dict('index'),

            # By WFH availability
            'by_wfh': df.groupby('wfh_available')['attrition_probability'].agg(['mean', 'count']).round(2).to_dict(
                'index'),

            # Risk distribution
            'risk_distribution': {
                'very_high': int((probabilities >= 50).sum()),
                'high': int(((probabilities >= 30) & (probabilities < 50)).sum()),
                'medium': int(((probabilities >= 15) & (probabilities < 30)).sum()),
                'low': int((probabilities < 15).sum())
            }
        }

        return jsonify(analytics)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/options/roles', methods=['GET'])
def get_available_roles():
    """Get list of available job roles"""
    return jsonify({
        'roles': list(label_encoders['role'].classes_)
    })


@app.route('/api/options/marital-statuses', methods=['GET'])
def get_marital_statuses():
    """Get list of available marital statuses"""
    return jsonify({
        'marital_statuses': list(label_encoders['marital_status'].classes_)
    })


@app.route('/api/options/all', methods=['GET'])
def get_all_options():
    """Get all dropdown options"""
    return jsonify({
        'roles': list(label_encoders['role'].classes_),
        'marital_statuses': list(label_encoders['marital_status'].classes_),
        'wfh_options': [
            {'value': 1, 'label': 'Yes'},
            {'value': 0, 'label': 'No'}
        ]
    })


@app.route('/api/predict/employee/<employee_id>/factors', methods=['GET'])
def get_employee_factors(employee_id):
    """Get detailed factor breakdown for an employee"""
    global last_uploaded_data

    if last_uploaded_data is None:
        return jsonify({'error': 'No data uploaded yet'}), 404

    try:
        df = pd.DataFrame(last_uploaded_data)
        employee = df[df['employee_id'] == employee_id]

        if employee.empty:
            return jsonify({'error': f'Employee {employee_id} not found'}), 404

        employee_data = employee.iloc[0].to_dict()

        # Calculate with factors
        result = calculate_attrition_with_factors(employee_data)

        # Format for frontend
        factors_data = [
            {
                'name': 'Salary Gap',
                'value': result['factors']['salary'],
                'description': 'Underpaid' if employee_data.get('salary_satisfaction', 0) < 0 else 'Well paid',
                'color': '#ef4444' if result['factors']['salary'] > 15 else '#f59e0b'
            },
            {
                'name': 'Economic Crisis',
                'value': result['factors']['economic'],
                'description': f"Impact: {employee_data.get('economic_crisis_impact', 0)}/10",
                'color': '#f97316'
            },
            {
                'name': 'COVID Impact',
                'value': result['factors']['covid'],
                'description': f"Score: {employee_data.get('covid_impact_score', 0)}/10",
                'color': '#8b5cf6'
            },
            {
                'name': 'Political Instability',
                'value': result['factors']['political'],
                'description': f"Concern: {employee_data.get('political_stability_concern', 0)}/10",
                'color': '#6366f1'
            },
            {
                'name': 'Job Factors',
                'value': result['factors']['job_factors'],
                'description': 'Role, WFH, Age, etc.',
                'color': '#3b82f6'
            }
        ]

        return jsonify({
            'employee_id': employee_id,
            'employee_data': employee_data,
            'attrition_probability': result['probability'],
            'factors': factors_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/single/factors', methods=['POST'])
def predict_single_with_factors():
    """Get prediction with factor breakdown for single employee"""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['age', 'time_at_current_role', 'marital_status',
                           'role', 'work_experience', 'wfh_available']

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Calculate with factors
        result = calculate_attrition_with_factors(data)

        # Format for frontend
        factors_data = [
            {
                'name': 'Salary Gap',
                'value': result['factors']['salary'],
                'description': 'Underpaid' if data.get('salary_satisfaction', 0) < 0 else 'Well paid',
                'color': '#ef4444' if result['factors']['salary'] > 15 else '#f59e0b'
            },
            {
                'name': 'Economic Crisis',
                'value': result['factors']['economic'],
                'description': f"Impact: {data.get('economic_crisis_impact', 0)}/10",
                'color': '#f97316'
            },
            {
                'name': 'COVID Impact',
                'value': result['factors']['covid'],
                'description': f"Score: {data.get('covid_impact_score', 0)}/10",
                'color': '#8b5cf6'
            },
            {
                'name': 'Political Instability',
                'value': result['factors']['political'],
                'description': f"Concern: {data.get('political_stability_concern', 0)}/10",
                'color': '#6366f1'
            },
            {
                'name': 'Job Factors',
                'value': result['factors']['job_factors'],
                'description': 'Role, WFH, Age, etc.',
                'color': '#3b82f6'
            }
        ]

        return jsonify({
            'employee_data': data,
            'attrition_probability': result['probability'],
            'factors': factors_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights/retention-strategies/<employee_id>', methods=['GET'])
def get_retention_strategies(employee_id):
    """Get personalized retention strategies for an employee"""
    global last_uploaded_data

    if last_uploaded_data is None:
        return jsonify({'error': 'No data uploaded yet'}), 404

    try:
        df = pd.DataFrame(last_uploaded_data)
        employee = df[df['employee_id'] == employee_id]

        if employee.empty:
            return jsonify({'error': f'Employee {employee_id} not found'}), 404

        employee_data = employee.iloc[0].to_dict()
        strategies = generate_retention_strategies(employee_data)

        return jsonify(strategies)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights/clustering', methods=['GET'])
def get_employee_clustering():
    """Get employee clustering analysis"""
    global last_uploaded_data

    if last_uploaded_data is None:
        return jsonify({'error': 'No data uploaded yet'}), 404

    try:
        result = perform_employee_clustering(last_uploaded_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/insights/cluster/<int:cluster_id>/strategies', methods=['GET'])
def get_cluster_strategies(cluster_id):
    """Get retention strategies for entire cluster"""
    global last_uploaded_data

    if last_uploaded_data is None:
        return jsonify({'error': 'No data uploaded yet'}), 404

    try:
        # Perform clustering
        clustering_result = perform_employee_clustering(last_uploaded_data)

        if 'error' in clustering_result:
            return jsonify(clustering_result), 400

        # Find the cluster
        cluster = next((c for c in clustering_result['clusters'] if c['id'] == cluster_id), None)

        if not cluster:
            return jsonify({'error': f'Cluster {cluster_id} not found'}), 404

        return jsonify({
            'cluster': cluster,
            'recommended_actions': cluster['recommended_actions'],
            'budget_estimate': len(cluster['top_employees']) * 15000,  # Rough estimate
            'expected_impact': f"Reduce cluster risk from {cluster['avg_risk']}% to ~{max(15, cluster['avg_risk'] - 20)}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("EMPLOYEE ATTRITION PREDICTION API")
    print("=" * 70)
    print("\nAvailable Endpoints:")
    print("  GET  /api/health                  - Health check")
    print("  POST /api/predict/single          - Predict single employee")
    print("  POST /api/predict/batch           - Predict multiple employees")
    print("  POST /api/predict/department      - Department analysis + forecasting")
    print("  POST /api/analytics/overview      - Get analytics insights")
    print("  GET  /api/options/roles           - Get available roles")
    print("  GET  /api/options/marital-statuses - Get marital statuses")
    print("  GET  /api/options/all             - Get all options")
    print("\nStarting Flask server...")
    print("=" * 70 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)