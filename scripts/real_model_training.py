import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the actual telco customer churn dataset"""
    print("=== Loading Telco Customer Churn Dataset ===")
    
    # In a real scenario, you would load from the CSV file
    # For this demo, we'll fetch the data from the URL
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/gBqE3R1cmOb0qyAv-nBNrbNUsg4tnYth6XlvMaQC0CYU6wn.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic info
        print("\n=== Dataset Overview ===")
        print(df.info())
        
        print("\n=== Target Variable Distribution ===")
        print(df['churned'].value_counts())
        print(f"Churn rate: {df['churned'].mean():.2%}")
        
        print("\n=== Missing Values ===")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found!")
            
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Comprehensive data preprocessing for telco churn data"""
    print("\n=== Data Preprocessing ===")
    
    df_processed = df.copy()
    
    # Handle TotalCharges - convert to numeric (some values might be strings)
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median())
    
    # Convert SeniorCitizen to string for consistency
    if 'SeniorCitizen' in df_processed.columns:
        df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(str)
    
    # Feature Engineering
    print("Creating new features...")
    
    # Tenure groups
    if 'tenure' in df_processed.columns:
        df_processed['tenure_group'] = pd.cut(
            pd.to_numeric(df_processed['tenure'], errors='coerce'),
            bins=[0, 12, 24, 48, 72],
            labels=['0-1_year', '1-2_years', '2-4_years', '4+_years']
        )
    
    # Monthly charges groups
    if 'MonthlyCharges' in df_processed.columns:
        df_processed['charges_group'] = pd.cut(
            pd.to_numeric(df_processed['MonthlyCharges'], errors='coerce'),
            bins=[0, 35, 65, 95, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
    
    # Total charges per tenure (spending rate)
    if 'TotalCharges' in df_processed.columns and 'tenure' in df_processed.columns:
        tenure_numeric = pd.to_numeric(df_processed['tenure'], errors='coerce')
        total_charges_numeric = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        df_processed['charges_per_tenure'] = total_charges_numeric / (tenure_numeric + 1)  # +1 to avoid division by zero
    
    # Count of additional services
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for col in service_columns:
        if col in df_processed.columns:
            df_processed[f'{col}_binary'] = (df_processed[col] == 'Yes').astype(int)
    
    # Total services count
    service_binary_cols = [f'{col}_binary' for col in service_columns if f'{col}_binary' in df_processed.columns]
    if service_binary_cols:
        df_processed['total_services'] = df_processed[service_binary_cols].sum(axis=1)
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    categorical_columns = [col for col in categorical_columns if col not in ['customerID', 'churned']]
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Prepare features and target
    feature_columns = [col for col in df_processed.columns if col not in ['customerID', 'churned']]
    X = df_processed[feature_columns]
    y = df_processed['churned']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, label_encoders, feature_columns

def train_and_evaluate_models(X, y, feature_names):
    """Train multiple models and select the best one"""
    print("\n=== Model Training and Evaluation ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Logistic Regression
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
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
        
        print(f"{name} Results:")
        print(f"  AUC-ROC: {auc_roc:.3f}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        if auc_roc > best_score:
            best_score = auc_roc
            best_model = model
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} with AUC-ROC: {best_score:.3f}")
    
    # Feature importance for the best model
    feature_importance = []
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        feature_importance = [
            {'feature': feat, 'importance': float(imp)} 
            for feat, imp in zip(feature_names, importance)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_[0])
        feature_importance = [
            {'feature': feat, 'importance': float(imp)} 
            for feat, imp in zip(feature_names, importance)
        ]
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return best_model, best_model_name, results, scaler, feature_importance

def main():
    """Main training pipeline"""
    print("🚀 Starting Real Telco Churn Model Training")
    print("=" * 50)
    
    # Load data
    df = load_and_explore_data()
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    # Preprocess data
    X, y, label_encoders, feature_names = preprocess_data(df)
    
    # Train models
    best_model, best_model_name, results, scaler, feature_importance = train_and_evaluate_models(X, y, feature_names)
    
    # Display final results
    print("\n" + "=" * 50)
    print("🎯 FINAL MODEL PERFORMANCE")
    print("=" * 50)
    
    best_results = results[best_model_name]
    print(f"Model Type: {best_model_name}")
    print(f"AUC-ROC Score: {best_results['auc_roc']:.3f}")
    print(f"Accuracy: {best_results['accuracy']:.3f}")
    print(f"Precision: {best_results['precision']:.3f}")
    print(f"Recall: {best_results['recall']:.3f}")
    print(f"F1-Score: {best_results['f1_score']:.3f}")
    print(f"Cross-Validation Score: {best_results['cv_score']:.3f} (+/- {best_results['cv_std']*2:.3f})")
    
    print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
    for i, feat in enumerate(feature_importance[:10], 1):
        print(f"{i:2d}. {feat['feature']:<25} {feat['importance']:.3f}")
    
    # Model insights
    print(f"\n💡 MODEL INSIGHTS:")
    print(f"• The model can distinguish between churners and non-churners with {best_results['auc_roc']:.1%} accuracy")
    print(f"• It correctly identifies {best_results['recall']:.1%} of actual churners")
    print(f"• {best_results['precision']:.1%} of customers predicted to churn actually do churn")
    print(f"• Overall prediction accuracy is {best_results['accuracy']:.1%}")
    
    print(f"\n✅ Model training completed successfully!")
    print(f"Ready for deployment and real-time predictions.")

if __name__ == "__main__":
    main()
