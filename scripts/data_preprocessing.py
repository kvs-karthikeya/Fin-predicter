import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load CSV data with error handling"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_data(df):
    """Perform basic data exploration"""
    print("=== Data Exploration ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    print("\nBasic statistics:")
    print(df.describe())
    
    # Check for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"\n{col} unique values: {df[col].nunique()}")
        if df[col].nunique() <= 10:
            print(df[col].value_counts())

def handle_missing_values(df, strategy='auto'):
    """Handle missing values in the dataset"""
    print("=== Handling Missing Values ===")
    
    df_clean = df.copy()
    
    for col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            print(f"Handling {missing_count} missing values in {col}")
            
            if df_clean[col].dtype == 'object':
                # For categorical variables
                if strategy == 'auto':
                    # Use mode (most frequent value)
                    mode_value = df_clean[col].mode()
                    fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                else:
                    fill_value = 'Unknown'
                df_clean[col] = df_clean[col].fillna(fill_value)
            else:
                # For numerical variables
                if strategy == 'auto':
                    # Use median for numerical variables
                    fill_value = df_clean[col].median()
                else:
                    fill_value = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_value)
    
    print("Missing values handled successfully!")
    return df_clean

def encode_categorical_variables(df, encoding_method='label'):
    """Encode categorical variables"""
    print("=== Encoding Categorical Variables ===")
    
    df_encoded = df.copy()
    encoders = {}
    
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        print(f"Encoding {col} using {encoding_method} encoding")
        
        if encoding_method == 'label':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
        elif encoding_method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
            encoders[col] = list(dummies.columns)
    
    print("Categorical encoding completed!")
    return df_encoded, encoders

def scale_numerical_features(df, method='standard'):
    """Scale numerical features"""
    print("=== Scaling Numerical Features ===")
    
    df_scaled = df.copy()
    numerical_columns = df_scaled.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    df_scaled[numerical_columns] = scaler.fit_transform(df_scaled[numerical_columns])
    
    print(f"Scaled {len(numerical_columns)} numerical features using {method} scaling")
    return df_scaled, scaler

def create_feature_engineering(df):
    """Create additional features from existing ones"""
    print("=== Feature Engineering ===")
    
    df_engineered = df.copy()
    
    # Example feature engineering for telecom data
    if 'tenure' in df_engineered.columns and 'monthly_charges' in df_engineered.columns:
        # Total amount spent
        df_engineered['total_spent'] = df_engineered['tenure'] * df_engineered['monthly_charges']
        print("Created 'total_spent' feature")
    
    if 'tenure' in df_engineered.columns:
        # Tenure categories
        df_engineered['tenure_group'] = pd.cut(
            df_engineered['tenure'], 
            bins=[0, 12, 24, 48, 72], 
            labels=['0-1 year', '1-2 years', '2-4 years', '4+ years']
        )
        print("Created 'tenure_group' feature")
    
    if 'monthly_charges' in df_engineered.columns:
        # Monthly charges categories
        df_engineered['charges_group'] = pd.cut(
            df_engineered['monthly_charges'],
            bins=[0, 35, 65, 95, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        print("Created 'charges_group' feature")
    
    print("Feature engineering completed!")
    return df_engineered

def preprocess_pipeline(file_path=None, df=None):
    """Complete preprocessing pipeline"""
    print("=== Starting Data Preprocessing Pipeline ===")
    
    # Load data
    if df is None:
        if file_path:
            df = load_data(file_path)
            if df is None:
                return None
        else:
            # Create synthetic data for demonstration
            print("Creating synthetic data for demonstration...")
            np.random.seed(42)
            n_samples = 1000
            
            df = pd.DataFrame({
                'customer_id': [f'C{i:04d}' for i in range(n_samples)],
                'tenure': np.random.randint(1, 73, n_samples),
                'monthly_charges': np.random.uniform(20, 120, n_samples),
                'total_charges': np.random.uniform(20, 8000, n_samples),
                'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
                'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
                'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
                'senior_citizen': np.random.choice([0, 1], n_samples),
                'partner': np.random.choice(['Yes', 'No'], n_samples),
                'dependents': np.random.choice(['Yes', 'No'], n_samples)
            })
            
            # Add some missing values for demonstration
            missing_indices = np.random.choice(n_samples, size=50, replace=False)
            df.loc[missing_indices, 'total_charges'] = np.nan
    
    # Explore data
    explore_data(df)
    
    # Handle missing values
    df_clean = handle_missing_values(df)
    
    # Feature engineering
    df_engineered = create_feature_engineering(df_clean)
    
    # Encode categorical variables
    df_encoded, encoders = encode_categorical_variables(df_engineered)
    
    # Scale numerical features
    df_final, scaler = scale_numerical_features(df_encoded)
    
    print("=== Preprocessing Pipeline Completed ===")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Final columns: {list(df_final.columns)}")
    
    return {
        'data': df_final,
        'original_data': df,
        'encoders': encoders,
        'scaler': scaler,
        'preprocessing_steps': [
            'missing_value_handling',
            'feature_engineering', 
            'categorical_encoding',
            'numerical_scaling'
        ]
    }

def main():
    """Main preprocessing function"""
    # Run preprocessing pipeline with synthetic data
    result = preprocess_pipeline()
    
    if result:
        print("\nPreprocessing completed successfully!")
        print("You can now use this preprocessed data for model training.")
        
        # Show sample of processed data
        print("\nSample of processed data:")
        print(result['data'].head())

if __name__ == "__main__":
    main()
