"""
Generate 100K Realistic Employee Dataset for Sri Lanka IT Industry
Based on real industry data and research

References:
- SLASSCOM HR Survey 2022
- Sri Lanka IT Industry Salary Survey 2023
- LinkedIn Workforce Sri Lanka 2022
- World Bank Sri Lanka Economic Crisis Report 2022
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# Sri Lankan IT industry data
ROLES = {
    'Trainee Developer': {'weight': 0.08, 'salary_range': (50000, 80000), 'min_exp': 0, 'max_exp': 1},
    'Junior Developer': {'weight': 0.15, 'salary_range': (80000, 150000), 'min_exp': 0, 'max_exp': 2},
    'Software Engineer': {'weight': 0.25, 'salary_range': (150000, 300000), 'min_exp': 1, 'max_exp': 5},
    'Senior Software Engineer': {'weight': 0.18, 'salary_range': (300000, 500000), 'min_exp': 4, 'max_exp': 10},
    'Tech Lead': {'weight': 0.12, 'salary_range': (450000, 700000), 'min_exp': 5, 'max_exp': 12},
    'Engineering Manager': {'weight': 0.08, 'salary_range': (600000, 1000000), 'min_exp': 6, 'max_exp': 15},
    'QA Engineer': {'weight': 0.06, 'salary_range': (120000, 250000), 'min_exp': 1, 'max_exp': 6},
    'Senior QA Engineer': {'weight': 0.04, 'salary_range': (250000, 400000), 'min_exp': 4, 'max_exp': 10},
    'DevOps Engineer': {'weight': 0.03, 'salary_range': (200000, 400000), 'min_exp': 2, 'max_exp': 8},
    'Architect': {'weight': 0.01, 'salary_range': (700000, 1200000), 'min_exp': 8, 'max_exp': 20},
}

DEPARTMENTS = {
    'Engineering': 0.70,
    'QA': 0.15,
    'DevOps': 0.10,
    'Business': 0.05
}

# Sri Lankan specific factors
SRI_LANKA_EVENTS = {
    'economic_crisis_2022': {
        'year': 2022,
        'impact_range': (7, 10),  # Very high impact
        'affected_period': (2022, 2023)
    },
    'covid_waves': {
        'wave1': {'year': 2020, 'months': (3, 8), 'impact': (6, 9)},
        'wave2': {'year': 2021, 'months': (1, 12), 'impact': (7, 10)},
        'wave3': {'year': 2022, 'months': (1, 4), 'impact': (5, 8)}
    }
}

# Political instability timeline
POLITICAL_EVENTS = {
    2019: (3, 5),  # Moderate
    2020: (4, 6),  # COVID + Political
    2021: (5, 7),  # Rising tensions
    2022: (8, 10),  # Crisis peak - Aragalaya protests
    2023: (6, 8),  # Recovery but unstable
    2024: (5, 7),  # Stabilizing
    2025: (4, 6)   # More stable
}


def calculate_market_salary(role, experience):
    """Calculate expected market salary based on role and experience"""
    role_info = ROLES[role]
    min_sal, max_sal = role_info['salary_range']

    # Experience factor (more experience = higher in range)
    exp_factor = min(1.0, experience / 10)  # Caps at 10 years

    # Calculate expected salary
    expected_salary = min_sal + (max_sal - min_sal) * exp_factor

    return expected_salary


def calculate_actual_salary(expected_salary, role, company_type='mid'):
    """Calculate actual salary with market variations"""

    # Company type variations
    company_multipliers = {
        'startup': np.random.uniform(0.7, 1.2),  # High variance
        'mid': np.random.uniform(0.85, 1.15),    # Moderate
        'large': np.random.uniform(0.95, 1.25),  # Better pay but less variance
        'multinational': np.random.uniform(1.1, 1.4)  # Best pay
    }

    multiplier = company_multipliers.get(company_type, 1.0)

    # Add random variation
    noise = np.random.uniform(-0.1, 0.1)

    actual_salary = expected_salary * multiplier * (1 + noise)

    return actual_salary


def calculate_salary_satisfaction(actual_salary, expected_salary):
    """Calculate salary satisfaction on -1 to 1 scale"""

    difference_ratio = (actual_salary - expected_salary) / expected_salary

    # Map to -1 to 1 scale
    # -30% underpaid = -1, +30% overpaid = +1
    satisfaction = np.clip(difference_ratio / 0.3, -1, 1)

    return satisfaction


def calculate_covid_impact(join_date, current_date=datetime(2025, 12, 1)):
    """Calculate COVID-19 impact based on joining date"""

    join_year = join_date.year
    join_month = join_date.month

    # Peak COVID periods
    if join_year == 2020 and join_month >= 3:
        return np.random.uniform(7, 10)  # High impact
    elif join_year == 2021:
        return np.random.uniform(7, 10)  # Sustained high impact
    elif join_year == 2022 and join_month <= 4:
        return np.random.uniform(5, 8)   # Moderate impact
    elif join_year == 2019:
        return np.random.uniform(4, 7)   # Witnessed pandemic
    else:
        return np.random.uniform(0, 3)   # Low/no impact


def calculate_economic_crisis_impact(join_date, age):
    """Calculate 2022 economic crisis impact"""

    join_year = join_date.year

    # 2022 crisis affected everyone present during that time
    if join_year <= 2022:
        # People who were working during crisis
        base_impact = np.random.uniform(7, 10)

        # Younger employees more affected (less savings)
        if age < 30:
            base_impact = min(10, base_impact * 1.2)

        return base_impact
    else:
        # Joined after crisis
        return np.random.uniform(1, 4)


def calculate_political_instability_concern(join_date, age):
    """Calculate political instability concern"""

    current_year = 2025

    # Get base concern for current year
    base_concern = np.random.uniform(*POLITICAL_EVENTS.get(current_year, (5, 7)))

    # Older employees more concerned about stability
    if age > 35:
        base_concern = min(10, base_concern * 1.1)

    # Recent joiners less attached, less concerned
    tenure_years = (datetime(2025, 12, 1) - join_date).days / 365
    if tenure_years < 1:
        base_concern *= 0.9

    return base_concern


def generate_realistic_employee_data(n_samples=100000):
    """Generate realistic employee dataset"""

    print(f"Generating {n_samples} realistic employee records...")
    print("Based on Sri Lankan IT industry data (2019-2025)")

    employees = []

    # Company type distribution
    company_types = ['startup', 'mid', 'large', 'multinational']
    company_weights = [0.25, 0.45, 0.20, 0.10]

    for i in range(n_samples):
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1}/{n_samples} records...")

        # Basic info
        employee_id = f"EMP{str(i+1).zfill(6)}"

        # Age (22-59, normal distribution around 30)
        age = int(np.clip(np.random.normal(32, 7), 22, 59))

        # Work experience (capped by age)
        max_possible_exp = age - 22
        work_experience = min(max_possible_exp, np.random.exponential(4))
        work_experience = round(work_experience, 1)

        # Role based on experience and age
        if work_experience < 1:
            possible_roles = ['Trainee Developer', 'Junior Developer']
        elif work_experience < 3:
            possible_roles = ['Junior Developer', 'Software Engineer', 'QA Engineer']
        elif work_experience < 5:
            possible_roles = ['Software Engineer', 'QA Engineer', 'DevOps Engineer']
        elif work_experience < 8:
            possible_roles = ['Senior Software Engineer', 'Senior QA Engineer', 'Tech Lead', 'DevOps Engineer']
        else:
            possible_roles = ['Tech Lead', 'Engineering Manager', 'Architect', 'Senior QA Engineer']

        role = random.choice(possible_roles)

        # Department based on role
        if 'QA' in role:
            department = 'QA'
        elif 'DevOps' in role:
            department = 'DevOps'
        elif 'Manager' in role or 'Architect' in role:
            department = 'Business'
        else:
            department = 'Engineering'

        # Time at current role (less than work experience)
        time_at_current_role = min(work_experience, np.random.exponential(2.5))
        time_at_current_role = round(time_at_current_role, 1)

        # Marital status (correlated with age)
        if age < 28:
            marital_status = np.random.choice(['Single', 'Married'], p=[0.75, 0.25])
        elif age < 35:
            marital_status = np.random.choice(['Single', 'Married'], p=[0.40, 0.60])
        else:
            marital_status = np.random.choice(['Single', 'Married'], p=[0.15, 0.85])

        # WFH availability (increased post-COVID)
        # 80% of companies now offer WFH
        wfh_available = np.random.choice([0, 1], p=[0.20, 0.80])

        # Company type
        company_type = np.random.choice(company_types, p=company_weights)

        # Joining date (between 2019-2025)
        days_range = (datetime(2025, 12, 1) - datetime(2019, 1, 1)).days
        random_days = random.randint(0, days_range)
        join_date = datetime(2019, 1, 1) + timedelta(days=random_days)

        # Salary calculation
        expected_salary = calculate_market_salary(role, work_experience)
        actual_salary = calculate_actual_salary(expected_salary, role, company_type)
        salary_satisfaction = calculate_salary_satisfaction(actual_salary, expected_salary)

        # External factors
        covid_impact_score = calculate_covid_impact(join_date)
        economic_crisis_impact = calculate_economic_crisis_impact(join_date, age)
        political_stability_concern = calculate_political_instability_concern(join_date, age)

        # Career growth perception (influenced by time in role)
        if time_at_current_role > 4:
            career_growth = np.random.uniform(2, 5)  # Stagnant
        elif time_at_current_role > 2.5:
            career_growth = np.random.uniform(4, 7)  # Moderate
        else:
            career_growth = np.random.uniform(6, 9)  # Good growth

        # Work-life balance (influenced by role and WFH)
        base_wlb = np.random.uniform(4, 8)
        if wfh_available:
            base_wlb = min(10, base_wlb + 1.5)
        if 'Manager' in role or 'Lead' in role:
            base_wlb *= 0.85  # Leaders work more
        work_life_balance = np.clip(base_wlb, 0, 10)

        # Manager relationship
        manager_relationship = np.random.uniform(4, 9)

        # Calculate attrition label (ground truth)
        # Based on multiple factors
        attrition_score = 0

        # Salary factor (30% weight)
        if salary_satisfaction < -0.3:
            attrition_score += 30
        elif salary_satisfaction < 0:
            attrition_score += 15

        # Career growth (25% weight)
        if career_growth < 4:
            attrition_score += 25
        elif career_growth < 6:
            attrition_score += 12

        # Work-life balance (15% weight)
        if work_life_balance < 4:
            attrition_score += 15
        elif work_life_balance < 6:
            attrition_score += 8

        # Manager relationship (10% weight)
        if manager_relationship < 4:
            attrition_score += 10
        elif manager_relationship < 6:
            attrition_score += 5

        # External factors (20% weight total)
        if economic_crisis_impact > 7:
            attrition_score += 10
        if covid_impact_score > 7:
            attrition_score += 5
        if political_stability_concern > 7:
            attrition_score += 5

        # WFH reduces attrition
        if not wfh_available:
            attrition_score += 8

        # Age factor (young employees more likely to leave)
        if age < 28:
            attrition_score += 5

        # Time in role (stagnation)
        if time_at_current_role > 4:
            attrition_score += 8

        # Add randomness
        attrition_score += np.random.uniform(-10, 10)

        # Convert to binary (50% threshold with noise)
        attrition_probability = min(95, max(5, attrition_score))
        will_leave = 1 if attrition_probability > 50 else 0

        # Add some randomness to make it realistic (not perfect prediction)
        if random.random() < 0.15:  # 15% noise
            will_leave = 1 - will_leave

        employees.append({
            'employee_id': employee_id,
            'age': age,
            'department': department,
            'role': role,
            'work_experience': work_experience,
            'time_at_current_role': time_at_current_role,
            'marital_status': marital_status,
            'wfh_available': wfh_available,
            'salary_satisfaction': round(salary_satisfaction, 3),
            'covid_impact_score': round(covid_impact_score, 2),
            'economic_crisis_impact': round(economic_crisis_impact, 2),
            'political_stability_concern': round(political_stability_concern, 2),
            'career_growth_opportunity': round(career_growth, 2),
            'work_life_balance': round(work_life_balance, 2),
            'manager_relationship': round(manager_relationship, 2),
            'company_type': company_type,
            'join_date': join_date.strftime('%Y-%m-%d'),
            'expected_salary': int(expected_salary),
            'actual_salary': int(actual_salary),
            'attrition': will_leave
        })

    print(f"\n✓ Generated {n_samples} employee records")

    df = pd.DataFrame(employees)

    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)

    print(f"\nAttrition Rate: {df['attrition'].mean()*100:.2f}%")
    print(f"  Will Leave: {df['attrition'].sum():,}")
    print(f"  Will Stay: {(len(df) - df['attrition'].sum()):,}")

    print(f"\nAge Distribution:")
    print(f"  Mean: {df['age'].mean():.1f}")
    print(f"  Range: {df['age'].min()} - {df['age'].max()}")

    print(f"\nExperience Distribution:")
    print(f"  Mean: {df['work_experience'].mean():.2f} years")
    print(f"  Range: {df['work_experience'].min():.1f} - {df['work_experience'].max():.1f} years")

    print(f"\nRole Distribution:")
    print(df['role'].value_counts())

    print(f"\nDepartment Distribution:")
    print(df['department'].value_counts())

    print(f"\nCompany Type Distribution:")
    print(df['company_type'].value_counts())

    print(f"\nMarital Status:")
    print(df['marital_status'].value_counts())

    print(f"\nWFH Available: {df['wfh_available'].sum():,} ({df['wfh_available'].mean()*100:.1f}%)")

    print(f"\nSalary Satisfaction:")
    print(f"  Mean: {df['salary_satisfaction'].mean():.3f}")
    print(f"  Underpaid (<0): {(df['salary_satisfaction'] < 0).sum():,}")
    print(f"  Fairly paid (±0.1): {((df['salary_satisfaction'] >= -0.1) & (df['salary_satisfaction'] <= 0.1)).sum():,}")
    print(f"  Overpaid (>0): {(df['salary_satisfaction'] > 0).sum():,}")

    print(f"\nExternal Factors (Mean Scores):")
    print(f"  COVID Impact: {df['covid_impact_score'].mean():.2f}/10")
    print(f"  Economic Crisis: {df['economic_crisis_impact'].mean():.2f}/10")
    print(f"  Political Concern: {df['political_stability_concern'].mean():.2f}/10")

    print(f"\nInternal Factors (Mean Scores):")
    print(f"  Career Growth: {df['career_growth_opportunity'].mean():.2f}/10")
    print(f"  Work-Life Balance: {df['work_life_balance'].mean():.2f}/10")
    print(f"  Manager Relationship: {df['manager_relationship'].mean():.2f}/10")

    print(f"\nSalary Statistics:")
    print(f"  Mean Expected: LKR {df['expected_salary'].mean():,.0f}")
    print(f"  Mean Actual: LKR {df['actual_salary'].mean():,.0f}")
    print(f"  Min Actual: LKR {df['actual_salary'].min():,.0f}")
    print(f"  Max Actual: LKR {df['actual_salary'].max():,.0f}")

    print("\n" + "="*70)

    return df


if __name__ == "__main__":
    # Generate 100K dataset
    df = generate_realistic_employee_data(100000)

    # Save to CSV
    output_file = '../datasets/employee_data_100k_realistic.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to {output_file}")

    # Also generate a smaller version for demo
    df_demo = df.sample(500, random_state=42)
    df_demo.to_csv('../test-data/employee_data_demo_500.csv', index=False)
    print(f"✓ Saved demo file: employee_data_demo_500.csv")