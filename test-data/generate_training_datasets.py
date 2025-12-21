import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

# Role distributions
roles = ['Trainee Developer', 'Junior Developer', 'Software Engineer',
         'Senior Software Engineer', 'Tech Lead', 'Engineering Manager',
         'QA Engineer', 'Senior QA Engineer', 'DevOps Engineer']
role_weights = [0.08, 0.15, 0.25, 0.18, 0.12, 0.08, 0.06, 0.04, 0.04]


def generate_base_employee_data(n_samples):
    """Generate core employee data"""
    data = []

    for i in range(1, n_samples + 1):
        age = np.random.randint(22, 58)
        work_experience = min(age - 22, np.random.exponential(5))
        time_at_current_role = min(work_experience, np.random.exponential(2.5))
        marital_status = np.random.choice(['Single', 'Married'], p=[0.4, 0.6])
        role = np.random.choice(roles, p=role_weights)
        wfh_available = np.random.choice([0, 1], p=[0.2, 0.8])

        data.append({
            'employee_id': f'EMP{i:04d}',
            'age': int(age),
            'work_experience': round(work_experience, 1),
            'time_at_current_role': round(time_at_current_role, 1),
            'marital_status': marital_status,
            'role': role,
            'wfh_available': wfh_available
        })

    return data


def calculate_attrition(row, factors_dict):
    """Calculate attrition based on core + dynamic factors"""
    attrition_score = 15  # Base score

    # Core factors
    if row['age'] < 28:
        attrition_score += 12
    elif row['age'] > 45:
        attrition_score += 8

    if row['marital_status'] == 'Single':
        attrition_score += 8

    if row['time_at_current_role'] > 4:
        attrition_score += 15
    elif row['time_at_current_role'] > 2.5:
        attrition_score += 8

    if row['wfh_available'] == 0:
        attrition_score += 10

    if row['work_experience'] < 2:
        attrition_score += 10
    elif row['work_experience'] > 10:
        attrition_score -= 5

    # Dynamic factors contribution
    for factor_name, weight in factors_dict.items():
        factor_value = row.get(factor_name, 5)
        attrition_score += (factor_value / 10) * weight

    # Add randomness
    attrition_score += np.random.uniform(-10, 10)

    # Threshold
    will_leave = 1 if attrition_score > 45 else 0

    # Add 15% label noise
    if np.random.random() < 0.15:
        will_leave = 1 - will_leave

    return will_leave


# ============= DATASET 1: Economic Crisis & Natural Disasters =============
print("=" * 70)
print("DATASET 1: Economic Crisis & Natural Disasters")
print("=" * 70)

train_data_1 = generate_base_employee_data(1500)
test_data_1 = generate_base_employee_data(2500)

# Add dynamic factors
for data_list in [train_data_1, test_data_1]:
    for row in data_list:
        # Economic crisis impact (0-10)
        row['economic_crisis_impact'] = round(np.random.uniform(4, 9), 1)

        # Flood impact (0-10, region-specific)
        row['flood_impact'] = round(np.random.uniform(0, 8), 1)

        # Drought impact (0-10)
        row['drought_impact'] = round(np.random.uniform(0, 6), 1)

# Calculate attrition
factors_1 = {
    'economic_crisis_impact': 20,
    'flood_impact': 15,
    'drought_impact': 10
}

for row in train_data_1:
    row['attrition'] = calculate_attrition(row, factors_1)

for row in test_data_1:
    row['attrition'] = calculate_attrition(row, factors_1)

df_train_1 = pd.DataFrame(train_data_1)
df_test_1 = pd.DataFrame(test_data_1)

df_train_1.to_csv('dataset1_economic_disasters_train_1500.csv', index=False)
df_test_1.to_csv('dataset1_economic_disasters_test_2500.csv', index=False)

print(f"✓ Train: {len(df_train_1)} samples, Attrition: {df_train_1['attrition'].mean() * 100:.1f}%")
print(f"✓ Test: {len(df_test_1)} samples, Attrition: {df_test_1['attrition'].mean() * 100:.1f}%")

# ============= DATASET 2: Health Crisis & Workplace Factors =============
print("\n" + "=" * 70)
print("DATASET 2: Health Crisis & Workplace Factors")
print("=" * 70)

train_data_2 = generate_base_employee_data(1500)
test_data_2 = generate_base_employee_data(2500)

for data_list in [train_data_2, test_data_2]:
    for row in data_list:
        # COVID impact
        exp = row['work_experience']
        row['covid_impact'] = round(7.5 if exp < 4 else 4.0, 1)

        # Workplace safety concern (0-10)
        row['workplace_safety_concern'] = round(np.random.uniform(2, 8), 1)

        # Mental health stress (0-10)
        row['mental_health_stress'] = round(np.random.uniform(3, 9), 1)

        # Commute difficulty (0-10)
        row['commute_difficulty'] = round(np.random.uniform(2, 8), 1) if row['wfh_available'] == 0 else round(
            np.random.uniform(0, 3), 1)

factors_2 = {
    'covid_impact': 12,
    'workplace_safety_concern': 10,
    'mental_health_stress': 18,
    'commute_difficulty': 8
}

for row in train_data_2:
    row['attrition'] = calculate_attrition(row, factors_2)

for row in test_data_2:
    row['attrition'] = calculate_attrition(row, factors_2)

df_train_2 = pd.DataFrame(train_data_2)
df_test_2 = pd.DataFrame(test_data_2)

df_train_2.to_csv('dataset2_health_workplace_train_1500.csv', index=False)
df_test_2.to_csv('dataset2_health_workplace_test_2500.csv', index=False)

print(f"✓ Train: {len(df_train_2)} samples, Attrition: {df_train_2['attrition'].mean() * 100:.1f}%")
print(f"✓ Test: {len(df_test_2)} samples, Attrition: {df_test_2['attrition'].mean() * 100:.1f}%")

# ============= DATASET 3: Political & Social Factors =============
print("\n" + "=" * 70)
print("DATASET 3: Political & Social Factors")
print("=" * 70)

train_data_3 = generate_base_employee_data(1500)
test_data_3 = generate_base_employee_data(2500)

for data_list in [train_data_3, test_data_3]:
    for row in data_list:
        # Political instability
        row['political_instability'] = round(np.random.uniform(4, 8), 1)

        # Social unrest impact
        row['social_unrest_impact'] = round(np.random.uniform(3, 7), 1)

        # Currency devaluation concern
        row['currency_devaluation'] = round(np.random.uniform(5, 9), 1)

        # Emigration consideration (0-10)
        if row['age'] < 35 and row['marital_status'] == 'Single':
            row['emigration_consideration'] = round(np.random.uniform(6, 10), 1)
        else:
            row['emigration_consideration'] = round(np.random.uniform(2, 6), 1)

factors_3 = {
    'political_instability': 12,
    'social_unrest_impact': 8,
    'currency_devaluation': 15,
    'emigration_consideration': 20
}

for row in train_data_3:
    row['attrition'] = calculate_attrition(row, factors_3)

for row in test_data_3:
    row['attrition'] = calculate_attrition(row, factors_3)

df_train_3 = pd.DataFrame(train_data_3)
df_test_3 = pd.DataFrame(test_data_3)

df_train_3.to_csv('dataset3_political_social_train_1500.csv', index=False)
df_test_3.to_csv('dataset3_political_social_test_2500.csv', index=False)

print(f"✓ Train: {len(df_train_3)} samples, Attrition: {df_train_3['attrition'].mean() * 100:.1f}%")
print(f"✓ Test: {len(df_test_3)} samples, Attrition: {df_test_3['attrition'].mean() * 100:.1f}%")

# ============= DATASET 4: Industry & Market Factors =============
print("\n" + "=" * 70)
print("DATASET 4: Industry & Market Factors")
print("=" * 70)

train_data_4 = generate_base_employee_data(1500)
test_data_4 = generate_base_employee_data(2500)

for data_list in [train_data_4, test_data_4]:
    for row in data_list:
        # Industry downturn
        row['industry_downturn'] = round(np.random.uniform(3, 7), 1)

        # Job market competitiveness (0-10, higher = more opportunities elsewhere)
        if row['role'] in ['Software Engineer', 'Senior Software Engineer', 'Tech Lead']:
            row['job_market_competitiveness'] = round(np.random.uniform(6, 10), 1)
        else:
            row['job_market_competitiveness'] = round(np.random.uniform(3, 7), 1)

        # Tech stack obsolescence concern
        if row['time_at_current_role'] > 3:
            row['tech_obsolescence_concern'] = round(np.random.uniform(5, 9), 1)
        else:
            row['tech_obsolescence_concern'] = round(np.random.uniform(2, 5), 1)

        # Startup ecosystem activity (0-10, higher = more startup opportunities)
        row['startup_ecosystem_activity'] = round(np.random.uniform(4, 8), 1)

factors_4 = {
    'industry_downturn': 10,
    'job_market_competitiveness': 18,
    'tech_obsolescence_concern': 12,
    'startup_ecosystem_activity': 8
}

for row in train_data_4:
    row['attrition'] = calculate_attrition(row, factors_4)

for row in test_data_4:
    row['attrition'] = calculate_attrition(row, factors_4)

df_train_4 = pd.DataFrame(train_data_4)
df_test_4 = pd.DataFrame(test_data_4)

df_train_4.to_csv('dataset4_industry_market_train_1500.csv', index=False)
df_test_4.to_csv('dataset4_industry_market_test_2500.csv', index=False)

print(f"✓ Train: {len(df_train_4)} samples, Attrition: {df_train_4['attrition'].mean() * 100:.1f}%")
print(f"✓ Test: {len(df_test_4)} samples, Attrition: {df_test_4['attrition'].mean() * 100:.1f}%")

# ============= DATASET 5: Compensation & Benefits Factors =============
print("\n" + "=" * 70)
print("DATASET 5: Compensation & Benefits Factors")
print("=" * 70)

train_data_5 = generate_base_employee_data(1500)
test_data_5 = generate_base_employee_data(2500)

for data_list in [train_data_5, test_data_5]:
    for row in data_list:
        # Salary competitiveness gap (0-10, higher = more underpaid)
        if row['role'] in ['Trainee Developer', 'Junior Developer']:
            row['salary_gap'] = round(np.random.uniform(5, 9), 1)
        elif row['role'] in ['Engineering Manager', 'Architect']:
            row['salary_gap'] = round(np.random.uniform(2, 5), 1)
        else:
            row['salary_gap'] = round(np.random.uniform(3, 7), 1)

        # Benefits inadequacy (0-10)
        row['benefits_inadequacy'] = round(np.random.uniform(3, 8), 1)

        # Pension/retirement concern (0-10)
        if row['age'] > 40:
            row['retirement_concern'] = round(np.random.uniform(6, 10), 1)
        else:
            row['retirement_concern'] = round(np.random.uniform(2, 5), 1)

        # Stock options attractiveness elsewhere (0-10)
        if row['role'] in ['Tech Lead', 'Engineering Manager', 'Senior Software Engineer']:
            row['stock_options_elsewhere'] = round(np.random.uniform(5, 9), 1)
        else:
            row['stock_options_elsewhere'] = round(np.random.uniform(2, 6), 1)

factors_5 = {
    'salary_gap': 25,
    'benefits_inadequacy': 12,
    'retirement_concern': 10,
    'stock_options_elsewhere': 15
}

for row in train_data_5:
    row['attrition'] = calculate_attrition(row, factors_5)

for row in test_data_5:
    row['attrition'] = calculate_attrition(row, factors_5)

df_train_5 = pd.DataFrame(train_data_5)
df_test_5 = pd.DataFrame(test_data_5)

df_train_5.to_csv('dataset5_compensation_benefits_train_1500.csv', index=False)
df_test_5.to_csv('dataset5_compensation_benefits_test_2500.csv', index=False)

print(f"✓ Train: {len(df_train_5)} samples, Attrition: {df_train_5['attrition'].mean() * 100:.1f}%")
print(f"✓ Test: {len(df_test_5)} samples, Attrition: {df_test_5['attrition'].mean() * 100:.1f}%")

# ============= SUMMARY =============
print("\n" + "=" * 70)
print("SUMMARY OF ALL DATASETS")
print("=" * 70)

datasets_info = [
    {
        'name': 'Dataset 1: Economic Crisis & Natural Disasters',
        'factors': ['economic_crisis_impact', 'flood_impact', 'drought_impact'],
        'train_file': 'dataset1_economic_disasters_train_1500.csv',
        'test_file': 'dataset1_economic_disasters_test_2500.csv'
    },
    {
        'name': 'Dataset 2: Health Crisis & Workplace',
        'factors': ['covid_impact', 'workplace_safety_concern', 'mental_health_stress', 'commute_difficulty'],
        'train_file': 'dataset2_health_workplace_train_1500.csv',
        'test_file': 'dataset2_health_workplace_test_2500.csv'
    },
    {
        'name': 'Dataset 3: Political & Social',
        'factors': ['political_instability', 'social_unrest_impact', 'currency_devaluation',
                    'emigration_consideration'],
        'train_file': 'dataset3_political_social_train_1500.csv',
        'test_file': 'dataset3_political_social_test_2500.csv'
    },
    {
        'name': 'Dataset 4: Industry & Market',
        'factors': ['industry_downturn', 'job_market_competitiveness', 'tech_obsolescence_concern',
                    'startup_ecosystem_activity'],
        'train_file': 'dataset4_industry_market_train_1500.csv',
        'test_file': 'dataset4_industry_market_test_2500.csv'
    },
    {
        'name': 'Dataset 5: Compensation & Benefits',
        'factors': ['salary_gap', 'benefits_inadequacy', 'retirement_concern', 'stock_options_elsewhere'],
        'train_file': 'dataset5_compensation_benefits_train_1500.csv',
        'test_file': 'dataset5_compensation_benefits_test_2500.csv'
    }
]

for i, ds in enumerate(datasets_info, 1):
    print(f"\n{i}. {ds['name']}")
    print(f"   Factors: {', '.join(ds['factors'])}")
    print(f"   Train: {ds['train_file']}")
    print(f"   Test:  {ds['test_file']}")

print("\n" + "=" * 70)
print("✓ All datasets generated successfully!")
print("=" * 70)