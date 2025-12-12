import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)


def generate_enhanced_dataset(n_records=200):
    """
    Generate dataset with contextual factors affecting attrition
    Based on research: COVID, economic crisis, salary gap, political stability
    """

    data = {
        'employee_id': [f'EMP{str(i).zfill(4)}' for i in range(1, n_records + 1)],
        'age': [],
        'time_at_current_role': [],
        'marital_status': [],
        'role': [],
        'work_experience': [],
        'wfh_available': [],

        # NEW CONTEXTUAL FACTORS
        'salary_satisfaction': [],  # -1 to 1 (negative = underpaid, positive = overpaid)
        'covid_impact_score': [],  # 0-10 (higher = more affected)
        'economic_crisis_impact': [],  # 0-10 (inflation, fuel crisis impact)
        'political_stability_concern': []  # 0-10 (political instability concern)
    }

    roles = [
        'Junior Developer', 'Software Engineer', 'Senior Software Engineer',
        'Tech Lead', 'Engineering Manager', 'QA Engineer', 'Senior QA Engineer',
        'DevOps Engineer', 'Business Analyst', 'Product Manager', 'Architect',
        'Director', 'Trainee Developer'
    ]

    # Expected salary ranges by role and experience
    salary_ranges = {
        'Trainee Developer': (50000, 80000),
        'Junior Developer': (60000, 120000),
        'QA Engineer': (80000, 150000),
        'Software Engineer': (110000, 200000),
        'DevOps Engineer': (120000, 220000),
        'Business Analyst': (100000, 180000),
        'Senior QA Engineer': (150000, 250000),
        'Senior Software Engineer': (180000, 350000),
        'Tech Lead': (250000, 450000),
        'Product Manager': (200000, 400000),
        'Architect': (300000, 550000),
        'Engineering Manager': (300000, 600000),
        'Director': (550000, 1200000)
    }

    for i in range(n_records):
        # Basic demographics
        age = np.random.randint(22, 60)
        data['age'].append(age)

        # Marital status
        if age < 26:
            marital = np.random.choice(['Single', 'Married'], p=[0.85, 0.15])
        elif age < 32:
            marital = np.random.choice(['Single', 'Married'], p=[0.45, 0.55])
        else:
            marital = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.20, 0.75, 0.05])
        data['marital_status'].append(marital)

        # Role based on age
        if age < 25:
            role = np.random.choice(['Junior Developer', 'Trainee Developer', 'QA Engineer'], p=[0.6, 0.3, 0.1])
        elif age < 30:
            role = np.random.choice(['Software Engineer', 'QA Engineer', 'DevOps Engineer'], p=[0.6, 0.25, 0.15])
        elif age < 35:
            role = np.random.choice(['Senior Software Engineer', 'Tech Lead', 'Senior QA Engineer'], p=[0.5, 0.3, 0.2])
        else:
            role = np.random.choice(
                ['Senior Software Engineer', 'Tech Lead', 'Engineering Manager', 'Architect', 'Director'],
                p=[0.3, 0.25, 0.2, 0.15, 0.1])
        data['role'].append(role)

        # Work experience
        max_exp = min(age - 22, 15)
        work_exp = round(np.random.exponential(scale=3.0), 2)
        work_exp = min(work_exp, max_exp)
        work_exp = max(0.1, work_exp)
        data['work_experience'].append(work_exp)

        # Time at current role
        time_role = round(min(work_exp, np.random.exponential(scale=2.0)), 2)
        time_role = max(0.1, time_role)
        data['time_at_current_role'].append(time_role)

        # WFH
        wfh = np.random.choice([0, 1], p=[0.25, 0.75])
        data['wfh_available'].append(wfh)

        # === NEW CONTEXTUAL FACTORS ===

        # 1. SALARY SATISFACTION (based on role, experience, and market)
        min_sal, max_sal = salary_ranges[role]
        expected_salary = min_sal + (max_sal - min_sal) * (work_exp / 15)  # Scale by experience
        actual_salary = expected_salary * np.random.uniform(0.7, 1.3)  # ±30% variation

        # Salary satisfaction: -1 (very underpaid) to +1 (very overpaid)
        salary_gap = (actual_salary - expected_salary) / expected_salary
        salary_satisfaction = np.clip(salary_gap, -1, 1)
        data['salary_satisfaction'].append(round(salary_satisfaction, 2))

        # 2. COVID IMPACT (2020-2022 had high impact)
        # Simulate joining year
        joining_year = 2024 - int(work_exp)
        if 2020 <= joining_year <= 2021:
            covid_impact = np.random.uniform(7, 10)  # High impact
        elif joining_year == 2022:
            covid_impact = np.random.uniform(4, 7)  # Medium impact
        else:
            covid_impact = np.random.uniform(0, 3)  # Low/no impact
        data['covid_impact_score'].append(round(covid_impact, 1))

        # 3. ECONOMIC CRISIS IMPACT (2022 peak crisis in Sri Lanka)
        # Higher for those who were working during 2022
        if joining_year <= 2022 and work_exp >= 2:
            # Was present during crisis
            economic_impact = np.random.uniform(7, 10)
        elif joining_year <= 2023:
            economic_impact = np.random.uniform(4, 7)
        else:
            economic_impact = np.random.uniform(0, 3)
        data['economic_crisis_impact'].append(round(economic_impact, 1))

        # 4. POLITICAL STABILITY CONCERN
        # Random distribution with slight bias toward concern
        political_concern = np.random.beta(2, 3) * 10  # Skewed distribution
        data['political_stability_concern'].append(round(political_concern, 1))

    df = pd.DataFrame(data)
    return df


# Generate and save
df = generate_enhanced_dataset(1145)
df.to_csv('demo_employees_enhanced_1000.csv', index=False)

print("✓ Generated enhanced dataset with contextual factors")
print(f"\nFirst 5 records:")
print(df.head().to_string())
print(f"\n✓ File saved: demo_employees_enhanced_200.csv")