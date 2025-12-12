"""
Main Attrition Prediction Model Training
Using 100K realistic Sri Lankan IT industry dataset

This is the primary model that predicts employee attrition
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

np.random.seed(42)


def load_and_prepare_data(filepath='../datasets/employee_data_100k_realistic.csv'):
    """Load and prepare the dataset"""

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(filepath)
    print(f"\n✓ Loaded {len(df):,} records from {filepath}")

    print(f"\nColumns: {list(df.columns)}")
    print(f"\nAttrition distribution:")
    print(df['attrition'].value_counts())
    print(f"Attrition rate: {df['attrition'].mean() * 100:.2f}%")

    return df


def prepare_features(df):
    """Prepare features for model training"""

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    # Create a copy
    df_model = df.copy()

    # Encode categorical variables
    label_encoders = {}

    categorical_cols = ['marital_status', 'role', 'department', 'company_type']

    for col in categorical_cols:
        le = LabelEncoder()
        df_model[f'{col}_encoded'] = le.fit_transform(df_model[col])
        label_encoders[col] = le
        print(f"✓ Encoded {col}: {len(le.classes_)} classes")

    # Feature selection for model
    feature_columns = [
        # Demographics
        'age',
        'work_experience',
        'time_at_current_role',
        'marital_status_encoded',

        # Job characteristics
        'role_encoded',
        'department_encoded',
        'company_type_encoded',
        'wfh_available',

        # Satisfaction factors
        'salary_satisfaction',
        'career_growth_opportunity',
        'work_life_balance',
        'manager_relationship',

        # External factors
        'covid_impact_score',
        'economic_crisis_impact',
        'political_stability_concern'
    ]

    X = df_model[feature_columns]
    y = df_model['attrition']

    print(f"\n✓ Selected {len(feature_columns)} features")
    print(f"\nFeatures: {feature_columns}")

    return X, y, label_encoders, feature_columns


def train_main_model(X, y, feature_columns):
    """Train the main attrition prediction model"""

    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    print(f"  Attrition in train: {y_train.mean() * 100:.2f}%")
    print(f"  Attrition in test: {y_test.mean() * 100:.2f}%")

    # Train Random Forest
    print("\n1. Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )

    rf_model.fit(X_train, y_train)
    print("   ✓ Random Forest trained")

    # Train Gradient Boosting
    print("\n2. Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    gb_model.fit(X_train, y_train)
    print("   ✓ Gradient Boosting trained")

    # Evaluate both models
    print("\n3. Evaluating models...")

    # Random Forest evaluation
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)

    print(f"\n   Random Forest:")
    print(f"   - Accuracy: {rf_accuracy:.4f} ({rf_accuracy * 100:.2f}%)")
    print(f"   - AUC-ROC: {rf_auc:.4f}")

    # Gradient Boosting evaluation
    gb_pred = gb_model.predict(X_test)
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]

    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_auc = roc_auc_score(y_test, gb_pred_proba)

    print(f"\n   Gradient Boosting:")
    print(f"   - Accuracy: {gb_accuracy:.4f} ({gb_accuracy * 100:.2f}%)")
    print(f"   - AUC-ROC: {gb_auc:.4f}")

    # Select best model
    if rf_auc > gb_auc:
        print(f"\n✓ Selected Random Forest (Better AUC: {rf_auc:.4f})")
        best_model = rf_model
        best_pred = rf_pred
        best_pred_proba = rf_pred_proba
        model_name = "RandomForest"
    else:
        print(f"\n✓ Selected Gradient Boosting (Better AUC: {gb_auc:.4f})")
        best_model = gb_model
        best_pred = gb_pred
        best_pred_proba = gb_pred_proba
        model_name = "GradientBoosting"

    # Detailed evaluation of best model
    print("\n4. Detailed Classification Report:")
    print(classification_report(y_test, best_pred,
                                target_names=['Will Stay', 'Will Leave'],
                                digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    print(f"\n5. Confusion Matrix:")
    print(f"   True Negatives (Correctly predicted stay): {cm[0, 0]:,}")
    print(f"   False Positives (Wrongly predicted leave): {cm[0, 1]:,}")
    print(f"   False Negatives (Wrongly predicted stay): {cm[1, 0]:,}")
    print(f"   True Positives (Correctly predicted leave): {cm[1, 1]:,}")

    # Feature importance
    print(f"\n6. Top 10 Most Important Features:")
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print(importance_df.head(10).to_string(index=False))

    # Cross-validation
    print(f"\n7. Cross-Validation (5-fold):")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5,
                                scoring='roc_auc', n_jobs=-1)
    print(f"   CV AUC Scores: {cv_scores}")
    print(f"   Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return best_model, best_pred, best_pred_proba, y_test, importance_df, model_name


def create_visualizations(model, y_test, y_pred, y_pred_proba, importance_df, model_name):
    """Create and save visualization plots"""

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Feature Importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top 15 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('main_model_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: main_model_feature_importance.png")

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Will Stay', 'Will Leave'],
                yticklabels=['Will Stay', 'Will Leave'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add percentages
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                     ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('main_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: main_model_confusion_matrix.png")

    # 3. ROC Curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('main_model_roc_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: main_model_roc_curve.png")

    # 4. Prediction Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6,
             label='Will Stay (Actual)', color='green', edgecolor='black')
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6,
             label='Will Leave (Actual)', color='red', edgecolor='black')
    plt.axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold (0.5)')
    plt.xlabel('Predicted Probability of Leaving', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('main_model_prediction_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: main_model_prediction_distribution.png")

    plt.close('all')

    print("\n✓ All visualizations created")


def save_model(model, label_encoders, feature_columns, model_name, metrics):
    """Save the trained model and metadata"""

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'feature_names': feature_columns,
        'model_type': model_name,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'version': '2.0'
    }

    filename = 'attrition_model_100k.pkl'
    joblib.dump(model_data, filename)
    print(f"✓ Model saved as '{filename}'")

    # Also save a smaller version compatible with existing system
    compatible_data = {
        'model': model,
        'label_encoders': {
            'marital_status': label_encoders['marital_status'],
            'role': label_encoders['role']
        },
        'feature_names': feature_columns
    }
    joblib.dump(compatible_data, 'attrition_model.pkl')
    print(f"✓ Compatible model saved as 'attrition_model.pkl'")

    return filename


def test_sample_predictions(model, label_encoders, feature_columns):
    """Test model with sample predictions"""

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    test_cases = [
        {
            'name': 'High Risk: Underpaid Junior, High Stress',
            'data': {
                'age': 26,
                'work_experience': 2.5,
                'time_at_current_role': 1.8,
                'marital_status': 'Single',
                'role': 'Software Engineer',
                'department': 'Engineering',
                'company_type': 'startup',
                'wfh_available': 0,
                'salary_satisfaction': -0.4,
                'career_growth_opportunity': 4.0,
                'work_life_balance': 4.5,
                'manager_relationship': 5.5,
                'covid_impact_score': 7.5,
                'economic_crisis_impact': 8.5,
                'political_stability_concern': 7.0
            }
        },
        {
            'name': 'Low Risk: Well-Paid Senior, Stable',
            'data': {
                'age': 38,
                'work_experience': 12.0,
                'time_at_current_role': 3.5,
                'marital_status': 'Married',
                'role': 'Tech Lead',
                'department': 'Engineering',
                'company_type': 'large',
                'wfh_available': 1,
                'salary_satisfaction': 0.3,
                'career_growth_opportunity': 7.5,
                'work_life_balance': 7.0,
                'manager_relationship': 8.0,
                'covid_impact_score': 3.0,
                'economic_crisis_impact': 5.5,
                'political_stability_concern': 5.0
            }
        },
        {
            'name': 'Medium Risk: Stagnant Mid-Level',
            'data': {
                'age': 32,
                'work_experience': 6.5,
                'time_at_current_role': 4.5,
                'marital_status': 'Married',
                'role': 'Senior Software Engineer',
                'department': 'Engineering',
                'company_type': 'mid',
                'wfh_available': 1,
                'salary_satisfaction': -0.1,
                'career_growth_opportunity': 4.5,
                'work_life_balance': 6.0,
                'manager_relationship': 6.5,
                'covid_impact_score': 5.0,
                'economic_crisis_impact': 7.0,
                'political_stability_concern': 6.0
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 60)

        # Prepare features
        data = test_case['data']
        features = []

        for col in feature_columns:
            if col == 'marital_status_encoded':
                features.append(label_encoders['marital_status'].transform([data['marital_status']])[0])
            elif col == 'role_encoded':
                features.append(label_encoders['role'].transform([data['role']])[0])
            elif col == 'department_encoded':
                features.append(label_encoders['department'].transform([data['department']])[0])
            elif col == 'company_type_encoded':
                features.append(label_encoders['company_type'].transform([data['company_type']])[0])
            else:
                features.append(data[col])

        X_test = np.array([features])

        # Predict
        pred = model.predict(X_test)[0]
        pred_proba = model.predict_proba(X_test)[0]

        print(f"   Prediction: {'Will LEAVE' if pred == 1 else 'Will STAY'}")
        print(f"   Probability of leaving: {pred_proba[1] * 100:.1f}%")
        print(f"   Confidence: {max(pred_proba) * 100:.1f}%")


def main():
    """Main training pipeline"""

    print("\n" + "=" * 70)
    print("MAIN ATTRITION PREDICTION MODEL TRAINING")
    print("100K Realistic Sri Lankan IT Industry Dataset")
    print("=" * 70)

    # Load data
    df = load_and_prepare_data()

    # Prepare features
    X, y, label_encoders, feature_columns = prepare_features(df)

    # Train model
    model, y_pred, y_pred_proba, y_test, importance_df, model_name = train_main_model(
        X, y, feature_columns
    )

    # Create visualizations
    create_visualizations(model, y_test, y_pred, y_pred_proba,
                          importance_df, model_name)

    # Save model
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'model_type': model_name,
        'n_samples': len(df),
        'n_features': len(feature_columns)
    }

    filename = save_model(model, label_encoders, feature_columns, model_name, metrics)

    # Test predictions
    test_sample_predictions(model, label_encoders, feature_columns)

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved as: {filename}")
    print(f"Model Type: {model_name}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"\nFiles created:")
    print("  - attrition_model_100k.pkl")
    print("  - attrition_model.pkl (compatible)")
    print("  - main_model_feature_importance.png")
    print("  - main_model_confusion_matrix.png")
    print("  - main_model_roc_curve.png")
    print("  - main_model_prediction_distribution.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()