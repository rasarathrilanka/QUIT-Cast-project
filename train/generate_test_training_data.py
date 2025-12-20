import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

print("Generating 1500 employee training dataset...")

# Role distributions
roles = ['Trainee Developer', 'Junior Developer', 'Software Engineer',
         'Senior Software Engineer', 'Tech Lead', 'Engineering Manager',
         'QA Engineer', 'Senior QA Engineer', 'DevOps Engineer']

role_weights = [0.08, 0.15, 0.25, 0.18, 0.12, 0.08, 0.06, 0.04, 0.04]

data = []

for i in range(1, 1501):
    # Basic demographics
    age = np.random.randint(22, 58)
    work_experience = min(age - 22, np.random.exponential(5))
    time_at_current_role = min(work_experience, np.random.exponential(2.5))
    marital_status = np.random.choice(['Single', 'Married'], p=[0.4, 0.6])
    role = np.random.choice(roles, p=role_weights)
    wfh_available = np.random.choice([0, 1], p=[0.2, 0.8])

    # Calculate attrition probability
    attrition_score = 15  # Base score

    # Age factor
    if age < 28:
        attrition_score += 12
    elif age > 45:
        attrition_score += 8

    # Marital status
    if marital_status == 'Single':
        attrition_score += 8

    # Time in role (stagnation)
    if time_at_current_role > 4:
        attrition_score += 15
    elif time_at_current_role > 2.5:
        attrition_score += 8

    # WFH
    if wfh_available == 0:
        attrition_score += 10

    # Experience level
    if work_experience < 2:
        attrition_score += 10  # Very junior, uncertain
    elif work_experience > 10:
        attrition_score -= 5  # Senior, more stable

    # Role-based
    if role in ['Trainee Developer', 'Junior Developer']:
        attrition_score += 8
    elif role in ['Engineering Manager', 'Tech Lead']:
        attrition_score -= 5

    # Add randomness
    attrition_score += np.random.uniform(-10, 10)

    # Convert to binary (with some noise)
    base_threshold = 45
    will_leave = 1 if attrition_score > base_threshold else 0

    # Add 15% label noise for realism
    if np.random.random() < 0.15:
        will_leave = 1 - will_leave

    data.append({
        'employee_id': f'EMP{i:04d}',
        'age': int(age),
        'work_experience': round(work_experience, 1),
        'time_at_current_role': round(time_at_current_role, 1),
        'marital_status': marital_status,
        'role': role,
        'wfh_available': wfh_available,
        'attrition': will_leave
    })

df = pd.DataFrame(data)

# Print statistics
print(f"\n✓ Generated {len(df)} records")
print(f"\nAttrition Distribution:")
print(f"  Will Leave (1): {df['attrition'].sum()} ({df['attrition'].mean() * 100:.1f}%)")
print(f"  Will Stay (0): {(df['attrition'] == 0).sum()} ({(1 - df['attrition'].mean()) * 100:.1f}%)")

print(f"\nAge Range: {df['age'].min()} - {df['age'].max()} (avg: {df['age'].mean():.1f})")
print(
    f"Experience Range: {df['work_experience'].min():.1f} - {df['work_experience'].max():.1f} (avg: {df['work_experience'].mean():.1f})")

print(f"\nRole Distribution:")
for role, count in df['role'].value_counts().items():
    print(f"  {role}: {count}")

print(f"\nMarital Status:")
for status, count in df['marital_status'].value_counts().items():
    print(f"  {status}: {count}")

print(f"\nWFH Distribution:")
print(f"  Has WFH: {(df['wfh_available'] == 1).sum()}")
print(f"  No WFH: {(df['wfh_available'] == 0).sum()}")

# Save
filename = 'training_data_1500.csv'
df.to_csv(filename, index=False)
print(f"\n✓ Saved to {filename}")