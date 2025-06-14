import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """Load and preprocess the training data"""
    print("Loading training data...")
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle missing values
    print("Handling missing values...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col != 'Churn':  # Don't encode target variable yet
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    if 'Churn' in df.columns:
        target_encoder = LabelEncoder()
        df['Churn'] = target_encoder.fit_transform(df['Churn'].astype(str))
        label_encoders['Churn'] = target_encoder
    
    return df, label_encoders

def train_models(X, y):
    """Train multiple models and select the best one"""
    print("Training multiple models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        if name == 'Logistic Regression':
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {
            'model': model,
            'auc_roc': auc_roc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name} - AUC-ROC: {auc_roc:.3f}, CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        if auc_roc > best_score:
            best_score = auc_roc
            best_model = model
            best_model_name = name
    
    return best_model, best_model_name, results, scaler

def get_feature_importance(model, feature_names, model_name):
    """Get feature importance from the trained model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return []
    
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return [{'feature': feat, 'importance': float(imp)} for feat, imp in feature_importance[:10]]

def main():
    """Main training pipeline"""
    print("=== Churn Prediction Model Training ===")
    
    # Load and preprocess data
    # Note: In a real scenario, you would load from an actual CSV file
    # For this demo, we'll create synthetic data
    print("Creating synthetic training data...")
    
    np.random.seed(42)
    n_samples = 5000
    
    # Create synthetic features
    data = {
        'tenure': np.random.randint(1, 73, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(20, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples),
        'partner': np.random.choice(['Yes', 'No'], n_samples),
        'dependents': np.random.choice(['Yes', 'No'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create synthetic target variable with some logic
    churn_prob = (
        (df['tenure'] < 12) * 0.3 +
        (df['monthly_charges'] > 80) * 0.2 +
        (df['contract_type'] == 'Month-to-month') * 0.3 +
        (df['tech_support'] == 'No') * 0.1 +
        np.random.uniform(0, 0.1, n_samples)
    )
    df['Churn'] = (churn_prob > 0.5).astype(int)
    
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    
    # Preprocess data
    df_processed, label_encoders = load_and_preprocess_data(None)  # Pass None since we're using synthetic data
    
    # For synthetic data, we'll do the preprocessing here
    categorical_columns = ['contract_type', 'payment_method', 'internet_service', 'tech_support', 'partner', 'dependents']
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Prepare features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    print(f"Features: {list(X.columns)}")
    print(f"Feature matrix shape: {X.shape}")
    
    # Train models
    best_model, best_model_name, results, scaler = train_models(X, y)
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best AUC-ROC score: {results[best_model_name]['auc_roc']:.3f}")
    
    # Get feature importance
    feature_importance = get_feature_importance(best_model, X.columns, best_model_name)
    
    print("\nTop 10 Most Important Features:")
    for i, feat_imp in enumerate(feature_importance[:10], 1):
        print(f"{i:2d}. {feat_imp['feature']:<20} {feat_imp['importance']:.3f}")
    
    # Save model and preprocessing components
    print("\nSaving model and preprocessing components...")
    
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': list(X.columns),
        'model_name': best_model_name,
        'metrics': results[best_model_name],
        'feature_importance': feature_importance
    }
    
    # In a real scenario, you would save these to files
    # joblib.dump(model_artifacts, 'churn_model.pkl')
    
    print("Model training completed successfully!")
    print(f"Final metrics:")
    print(f"  AUC-ROC: {results[best_model_name]['auc_roc']:.3f}")
    print(f"  Accuracy: {results[best_model_name]['accuracy']:.3f}")
    print(f"  Precision: {results[best_model_name]['precision']:.3f}")
    print(f"  Recall: {results[best_model_name]['recall']:.3f}")
    print(f"  F1-Score: {results[best_model_name]['f1_score']:.3f}")
    print(f"  CV Score: {results[best_model_name]['cv_score']:.3f} (+/- {results[best_model_name]['cv_std']*2:.3f})")

if __name__ == "__main__":
    main()
