"""
Attrition Prediction Interface
Simple interface to predict employee attrition using the trained model
"""

import joblib
import pandas as pd
import numpy as np


class AttritionPredictionAPI:
    """
    Simple API for making attrition predictions
    """

    def __init__(self, model_path='attrition_model.pkl'):
        """Load the trained model"""
        print("Loading trained model...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print("✓ Model loaded successfully\n")

    def predict_single_employee(self, age, time_at_current_role, marital_status,
                                role, work_experience, wfh_available):
        """
        Predict attrition probability for a single employee

        Parameters:
        -----------
        age : int
            Employee age (22-60)
        time_at_current_role : float
            Years in current role
        marital_status : str
            'Single', 'Married', or 'Divorced'
        role : str
            Job role (e.g., 'Software Engineer', 'Senior Software Engineer', etc.)
        work_experience : float
            Total years at company
        wfh_available : int
            1 if WFH available, 0 if not

        Returns:
        --------
        dict : Prediction results including probability and risk level
        """
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'age': age,
            'time_at_current_role': time_at_current_role,
            'marital_status': marital_status,
            'role': role,
            'work_experience': work_experience,
            'wfh_available': wfh_available
        }])

        # Encode categorical variables
        input_data['marital_status_encoded'] = self.label_encoders['marital_status'].transform(
            input_data['marital_status']
        )
        input_data['role_encoded'] = self.label_encoders['role'].transform(
            input_data['role']
        )

        # Select features
        X = input_data[self.feature_names]

        # Predict
        attrition_prob = self.model.predict_proba(X)[0, 1]
        prediction = self.model.predict(X)[0]

        # Determine risk level
        if attrition_prob < 0.15:
            risk_level = "Low Risk"
            risk_color = "🟢"
        elif attrition_prob < 0.30:
            risk_level = "Medium Risk"
            risk_color = "🟡"
        elif attrition_prob < 0.50:
            risk_level = "High Risk"
            risk_color = "🟠"
        else:
            risk_level = "Very High Risk"
            risk_color = "🔴"

        return {
            'attrition_probability': round(attrition_prob * 100, 2),
            'will_leave': bool(prediction),
            'risk_level': f"{risk_color} {risk_level}",
            'confidence': 'High' if abs(attrition_prob - 0.5) > 0.3 else 'Medium'
        }

    def predict_batch(self, employee_df):
        """
        Predict attrition for multiple employees

        Parameters:
        -----------
        employee_df : DataFrame
            DataFrame with columns: age, time_at_current_role, marital_status,
            role, work_experience, wfh_available

        Returns:
        --------
        DataFrame : Original data with prediction columns added
        """
        df = employee_df.copy()

        # Encode categorical variables
        df['marital_status_encoded'] = self.label_encoders['marital_status'].transform(
            df['marital_status']
        )
        df['role_encoded'] = self.label_encoders['role'].transform(df['role'])

        # Select features
        X = df[self.feature_names]

        # Predict
        df['attrition_probability'] = self.model.predict_proba(X)[:, 1] * 100
        df['predicted_attrition'] = self.model.predict(X)

        # Add risk levels
        df['risk_level'] = df['attrition_probability'].apply(self._get_risk_level)

        return df

    def _get_risk_level(self, prob):
        """Convert probability to risk level"""
        if prob < 15:
            return "🟢 Low Risk"
        elif prob < 30:
            return "🟡 Medium Risk"
        elif prob < 50:
            return "🟠 High Risk"
        else:
            return "🔴 Very High Risk"

    def calculate_department_attrition_rate(self, employee_df):
        """
        Calculate expected attrition rate for a department/company

        Parameters:
        -----------
        employee_df : DataFrame
            DataFrame with employee data

        Returns:
        --------
        dict : Attrition statistics
        """
        predictions = self.predict_batch(employee_df)

        total_employees = len(predictions)
        expected_leavers = predictions['predicted_attrition'].sum()
        avg_attrition_prob = predictions['attrition_probability'].mean()

        # Count by risk level
        risk_counts = predictions['risk_level'].value_counts()

        return {
            'total_employees': total_employees,
            'expected_leavers': int(expected_leavers),
            'expected_attrition_rate': round(avg_attrition_prob, 2),
            'risk_distribution': risk_counts.to_dict(),
            'high_risk_employees': len(predictions[predictions['attrition_probability'] >= 50])
        }

    def get_available_roles(self):
        """Get list of available job roles"""
        return list(self.label_encoders['role'].classes_)

    def get_available_marital_statuses(self):
        """Get list of available marital statuses"""
        return list(self.label_encoders['marital_status'].classes_)


def main():
    """Demonstration of the prediction API"""
    print("=" * 70)
    print("ATTRITION PREDICTION API - DEMONSTRATION")
    print("=" * 70)
    print()

    # Initialize API
    api = AttritionPredictionAPI()

    # Show available options
    print("Available Job Roles:")
    for role in api.get_available_roles():
        print(f"  • {role}")

    print("\nAvailable Marital Statuses:")
    for status in api.get_available_marital_statuses():
        print(f"  • {status}")

    # Example 1: Single prediction
    print("\n" + "=" * 70)
    print("EXAMPLE 1: SINGLE EMPLOYEE PREDICTION")
    print("=" * 70)

    result = api.predict_single_employee(
        age=28,
        time_at_current_role=1.5,
        marital_status='Single',
        role='Software Engineer',
        work_experience=3.0,
        wfh_available=1
    )

    print("\nEmployee Profile:")
    print("  Age: 28 years")
    print("  Role: Software Engineer")
    print("  Work Experience: 3.0 years")
    print("  Time in Current Role: 1.5 years")
    print("  Marital Status: Single")
    print("  WFH Available: Yes")

    print("\nPrediction Results:")
    print(f"  Attrition Probability: {result['attrition_probability']}%")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Will Leave?: {'Yes' if result['will_leave'] else 'No'}")
    print(f"  Confidence: {result['confidence']}")

    # Example 2: Batch prediction
    print("\n" + "=" * 70)
    print("EXAMPLE 2: DEPARTMENT/TEAM PREDICTION")
    print("=" * 70)

    # Create sample department data
    department_data = pd.DataFrame([
        {'age': 25, 'time_at_current_role': 0.8, 'marital_status': 'Single',
         'role': 'Junior Developer', 'work_experience': 1.2, 'wfh_available': 0},
        {'age': 30, 'time_at_current_role': 2.5, 'marital_status': 'Married',
         'role': 'Software Engineer', 'work_experience': 4.5, 'wfh_available': 1},
        {'age': 35, 'time_at_current_role': 3.0, 'marital_status': 'Married',
         'role': 'Senior Software Engineer', 'work_experience': 7.0, 'wfh_available': 1},
        {'age': 27, 'time_at_current_role': 5.0, 'marital_status': 'Single',
         'role': 'QA Engineer', 'work_experience': 5.5, 'wfh_available': 1},
        {'age': 40, 'time_at_current_role': 2.0, 'marital_status': 'Married',
         'role': 'Tech Lead', 'work_experience': 8.0, 'wfh_available': 1},
    ])

    print(f"\nAnalyzing department with {len(department_data)} employees...")

    # Get predictions
    predictions = api.predict_batch(department_data)

    # Display results
    print("\nEmployee-by-Employee Breakdown:")
    print("-" * 70)
    for idx, row in predictions.iterrows():
        print(f"{idx + 1}. {row['role']:30s} | "
              f"Prob: {row['attrition_probability']:5.1f}% | "
              f"{row['risk_level']}")

    # Calculate department statistics
    dept_stats = api.calculate_department_attrition_rate(department_data)

    print("\n" + "-" * 70)
    print("DEPARTMENT ATTRITION FORECAST")
    print("-" * 70)
    print(f"Total Employees: {dept_stats['total_employees']}")
    print(f"Expected Leavers: {dept_stats['expected_leavers']}")
    print(f"Expected Attrition Rate: {dept_stats['expected_attrition_rate']}%")
    print(f"High-Risk Employees: {dept_stats['high_risk_employees']}")

    print("\nRisk Distribution:")
    for risk, count in sorted(dept_stats['risk_distribution'].items()):
        print(f"  {risk}: {count} employee(s)")

    # Example 3: Interactive prediction
    print("\n" + "=" * 70)
    print("EXAMPLE 3: CUSTOM PREDICTION")
    print("=" * 70)

    custom_result = api.predict_single_employee(
        age=33,
        time_at_current_role=4.5,
        marital_status='Single',
        role='Software Engineer',
        work_experience=6.5,
        wfh_available=0
    )

    print("\nScenario: Long-tenured employee without promotion, no WFH")
    print("  Age: 33, Role: Software Engineer")
    print("  Work Experience: 6.5 years, Time in Role: 4.5 years")
    print("  Marital: Single, WFH: No")
    print(f"\n  → Attrition Risk: {custom_result['attrition_probability']}%")
    print(f"  → {custom_result['risk_level']}")

    print("\n" + "=" * 70)
    print("API DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nThe model is ready for production use!")
    print("You can integrate this API into your HR system.")


if __name__ == "__main__":
    main()