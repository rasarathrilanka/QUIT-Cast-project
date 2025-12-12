"""
QUICK START SCRIPT - Employee Attrition Prediction
====================================================

This script demonstrates how to quickly use the attrition prediction model.
Simply modify the employee data below and run this script.

"""

import sys

from prediction_api import AttritionPredictionAPI
import pandas as pd


def predict_single_employee_example():
    """Example: Predict attrition for a single employee"""

    print("=" * 70)
    print("SINGLE EMPLOYEE PREDICTION")
    print("=" * 70)

    # Initialize the API
    api = AttritionPredictionAPI()

    # MODIFY THESE VALUES for your employee
    employee_data = {
        'age': 29,
        'time_at_current_role': 2.0,
        'marital_status': 'Single',  # Options: 'Single', 'Married', 'Divorced'
        'role': 'Software Engineer',  # See list of roles below
        'work_experience': 3.5,
        'wfh_available': 1  # 1 = Yes, 0 = No
    }

    # Make prediction
    result = api.predict_single_employee(**employee_data)

    # Display results
    print("\nEmployee Profile:")
    print(f"  Age: {employee_data['age']}")
    print(f"  Role: {employee_data['role']}")
    print(f"  Work Experience: {employee_data['work_experience']} years")
    print(f"  Time in Current Role: {employee_data['time_at_current_role']} years")
    print(f"  Marital Status: {employee_data['marital_status']}")
    print(f"  WFH Available: {'Yes' if employee_data['wfh_available'] else 'No'}")

    print("\n" + "-" * 70)
    print("PREDICTION RESULTS")
    print("-" * 70)
    print(f"  Attrition Probability: {result['attrition_probability']}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Will Likely Leave?: {'Yes' if result['will_leave'] else 'No'}")
    print(f"  Prediction Confidence: {result['confidence']}")
    print()


def predict_team_example():
    """Example: Predict attrition for a team/department"""

    print("=" * 70)
    print("TEAM/DEPARTMENT PREDICTION")
    print("=" * 70)

    # Initialize the API
    api = AttritionPredictionAPI()

    # MODIFY THIS DATA for your team
    team_data = pd.DataFrame([
        # Employee 1
        {
            'age': 25,
            'time_at_current_role': 0.5,
            'marital_status': 'Single',
            'role': 'Junior Developer',
            'work_experience': 1.0,
            'wfh_available': 0
        },
        # Employee 2
        {
            'age': 30,
            'time_at_current_role': 2.0,
            'marital_status': 'Married',
            'role': 'Software Engineer',
            'work_experience': 4.0,
            'wfh_available': 1
        },
        # Employee 3
        {
            'age': 35,
            'time_at_current_role': 3.5,
            'marital_status': 'Married',
            'role': 'Senior Software Engineer',
            'work_experience': 7.0,
            'wfh_available': 1
        },
        # Employee 4
        {
            'age': 28,
            'time_at_current_role': 5.0,
            'marital_status': 'Single',
            'role': 'QA Engineer',
            'work_experience': 5.5,
            'wfh_available': 1
        },
        # Employee 5
        {
            'age': 40,
            'time_at_current_role': 2.0,
            'marital_status': 'Married',
            'role': 'Tech Lead',
            'work_experience': 10.0,
            'wfh_available': 1
        },
        # Add more employees here...
    ])

    print(f"\nAnalyzing team with {len(team_data)} employees...")

    # Get predictions
    predictions = api.predict_batch(team_data)

    # Display individual results
    print("\n" + "-" * 70)
    print("INDIVIDUAL EMPLOYEE PREDICTIONS")
    print("-" * 70)
    for idx, row in predictions.iterrows():
        print(f"\nEmployee {idx + 1}:")
        print(f"  Role: {row['role']}")
        print(f"  Age: {row['age']}, Experience: {row['work_experience']} years")
        print(f"  Attrition Probability: {row['attrition_probability']:.1f}%")
        print(f"  Risk Level: {row['risk_level']}")

    # Calculate team statistics
    dept_stats = api.calculate_department_attrition_rate(team_data)

    # Display team summary
    print("\n" + "-" * 70)
    print("TEAM ATTRITION FORECAST")
    print("-" * 70)
    print(f"  Total Employees: {dept_stats['total_employees']}")
    print(f"  Expected Leavers: {dept_stats['expected_leavers']}")
    print(f"  Expected Attrition Rate: {dept_stats['expected_attrition_rate']}%")
    print(f"  High-Risk Employees: {dept_stats['high_risk_employees']}")

    print("\n  Risk Distribution:")
    for risk, count in sorted(dept_stats['risk_distribution'].items()):
        print(f"    {risk}: {count} employee(s)")
    print()


def show_available_options():
    """Display available job roles and marital statuses"""

    print("=" * 70)
    print("AVAILABLE OPTIONS")
    print("=" * 70)

    api = AttritionPredictionAPI()

    print("\nSupported Job Roles:")
    for i, role in enumerate(api.get_available_roles(), 1):
        print(f"  {i:2d}. {role}")

    print("\nSupported Marital Statuses:")
    for i, status in enumerate(api.get_available_marital_statuses(), 1):
        print(f"  {i}. {status}")
    print()


def main():
    """Main execution"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  EMPLOYEE ATTRITION PREDICTION - QUICK START".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    # Show available options
    show_available_options()

    # Run single employee example
    predict_single_employee_example()

    # Run team example
    predict_team_example()

    print("=" * 70)
    print("QUICK START COMPLETE")
    print("=" * 70)
    print()
    print("To use this model:")
    print("1. Modify the employee_data or team_data in this script")
    print("2. Run: python quickstart.py")
    print("3. View the prediction results")
    print()
    print("For more details, see MODEL_DOCUMENTATION.md")
    print()


if __name__ == "__main__":
    main()