"""
Machine Learning Training Pipeline Core for the Universal Risk Predictor.
Handles feature detection, scaling, imputation, random forest classifier fitting,
evaluation, metadata mapping, and serialization.
"""

import os
import sys
from typing import Any, Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

# Import centralized configuration constants
from config import (
    DEFAULT_DATASET_PATH,
    MODEL_OUTPUT_PATH,
    TARGET_COLUMN_FALLBACKS,
    COLUMN_MAPPING
)

# Ensure the output models folder structure exists natively
os.makedirs("models", exist_ok=True)

def detect_column_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Dynamically infers numerical and categorical columns for machine learning.
    Excludes unique IDs and date/time columns to prevent overfitting.
    
    Args:
        df: Input raw pandas DataFrame.
        target_col: The label or target class column name.
        
    Returns:
        A tuple of (numerical_features, categorical_features, ignored_features) lists.
    """
    numerical_features: List[str] = []
    categorical_features: List[str] = []
    ignored_features: List[str] = []
    
    for col in df.columns:
        if col == target_col:
            continue
            
        col_lower = col.lower()
        # Common ID patterns
        if col_lower in ['order_id', 'product_id', 'user_id', 'customer_id', 'id', 'uuid', 'guid', 'serial']:
            ignored_features.append(col)
            continue
            
        # Ignore datetime objects directly, as they cannot be numeric scaled easily
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            ignored_features.append(col)
            continue
            
        # Treat object/categorical/string datatypes as categorical features
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            # Check for high cardinality string ID columns
            n_unique = df[col].nunique()
            if n_unique > 100 and n_unique > 0.5 * len(df):
                ignored_features.append(col)
                continue
            categorical_features.append(col)
        else:
            # Numeric column
            n_unique = df[col].nunique()
            # If numeric has 1 unique value and it's basically the record number
            if n_unique == len(df) and df[col].dtype == np.integer:
                ignored_features.append(col)
                continue
            numerical_features.append(col)
            
    return numerical_features, categorical_features, ignored_features

def train_custom_model(
    df: pd.DataFrame, 
    target_col: str, 
    numerical_features: List[str], 
    categorical_features: List[str]
) -> Dict[str, Any]:
    """
    Builds and trains a scikit-learn pipeline for classification.
    Calculates detailed metrics, feature importances, and metadata for dynamic UI generation.
    
    Args:
        df: Training DataFrame.
        target_col: Binary target label column.
        numerical_features: List of numerical feature names.
        categorical_features: List of categorical feature names.
        
    Returns:
        A dictionary containing the pipeline, features, metrics, and feature metadata.
    """
    # Defensive downsampling for large datasets to prevent OutOfMemory (OOM) crashes in the UI
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        
    # 1. Clean target column: handle NaNs and convert to binary
    df = df.dropna(subset=[target_col])
    y = df[target_col]
    
    if y.dtype == object or y.dtype == str:
        # Convert yes/no, returned/not returned to 1/0
        y = y.str.lower().str.strip().isin(['returned', '1', 'yes', 'true', 'rto', 'returned']).astype(int)
    else:
        y = y.astype(int)
        
    X = df[numerical_features + categorical_features]
    
    # 2. Formulate Preprocessing Transformers Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Standardize sparse parameter depending on scikit-learn version
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', encoder)
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ])
        
    # 3. Synthesize End-to-End Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    
    # 4. Segment Split Parameters (80% Fit Runway, 20% Evaluation Ground)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit pipeline
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model_pipeline.predict(X_test)
    try:
        y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        y_pred_proba = None
        
    accuracy = accuracy_score(y_test, y_pred)
    
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            roc_auc = 0.5
    else:
        roc_auc = 0.5
        
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate feature importances mapping
    classifier = model_pipeline.named_steps['classifier']
    importances = classifier.feature_importances_
    
    # Get feature names out
    try:
        feature_names = model_pipeline.named_steps['preprocessing'].get_feature_names_out()
    except Exception:
        feature_names = []
        for col in numerical_features:
            feature_names.append(col)
        try:
            ohe = model_pipeline.named_steps['preprocessing'].named_transformers_['cat'].named_steps['encoder']
            for i, col in enumerate(categorical_features):
                cats = ohe.categories_[i]
                for cat in cats:
                    feature_names.append(f"{col}_{cat}")
        except Exception:
            for col in categorical_features:
                feature_names.append(col)
                
    # Build dictionary of detailed feature importances
    importances_dict = {}
    if len(feature_names) == len(importances):
        for name, imp in zip(feature_names, importances):
            importances_dict[name] = float(imp)
    else:
        # Fallback
        for idx, imp in enumerate(importances):
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            importances_dict[name] = float(imp)
            
    # Calculate aggregated importances for original features
    agg_importances = {}
    for col in numerical_features + categorical_features:
        agg_importances[col] = 0.0
        
    for k, v in importances_dict.items():
        clean_k = k
        if k.startswith("num__"):
            clean_k = k[5:]
        elif k.startswith("cat__"):
            clean_k = k[5:]
            
        for col in numerical_features:
            if clean_k == col:
                agg_importances[col] += v
                break
        for col in categorical_features:
            if clean_k.startswith(col + "_") or clean_k == col:
                agg_importances[col] += v
                break
        
    # Standardize aggregate importances to sum to 1.0
    agg_sum = sum(agg_importances.values())
    if agg_sum > 0:
        agg_importances = {k: v / agg_sum for k, v in agg_importances.items()}
        
    # Calculate default/median values for dynamic UI forms
    feature_metadata = {}
    for col in numerical_features:
        feature_metadata[col] = {
            'type': 'numerical',
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else 0.0,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else 100.0,
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0,
            'median': float(df[col].median()) if not pd.isna(df[col].median()) else 0.0,
        }
    for col in categorical_features:
        # Get unique category lists (excluding missing values)
        unique_vals = [str(x) for x in df[col].dropna().unique()]
        unique_vals = sorted(unique_vals)
        if len(unique_vals) == 0:
            unique_vals = ['Unknown']
        feature_metadata[col] = {
            'type': 'categorical',
            'unique_values': unique_vals,
            'mode': str(df[col].mode()[0]) if (not df[col].mode().empty) else unique_vals[0]
        }
        
    # Compile bundle
    model_metadata = {
        'pipeline': model_pipeline,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'target_column': target_col,
        'feature_metadata': feature_metadata,
        'feature_importances': importances_dict,
        'aggregated_importances': agg_importances,
        'metrics': {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'classification_report': report
        }
    }
    
    # Save model metadata bundle
    joblib.dump(model_metadata, MODEL_OUTPUT_PATH)
    print(f"[SUCCESS] Model architecture and metadata saved to '{MODEL_OUTPUT_PATH}'")
    return model_metadata

if __name__ == '__main__':
    # 1. Access the dataset path location
    file_path = DEFAULT_DATASET_PATH
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing base training data array at target location: '{file_path}'")
        
    print(f"[INFO] Ingesting transactional logs from: {file_path}")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path, engine="openpyxl")
        
    # Standardize columns mapping using configuration mapping dict
    df = df.rename(columns=lambda x: COLUMN_MAPPING.get(x, x))
    df = df.rename(columns=lambda x: COLUMN_MAPPING.get(x.strip(), x))
    df.columns = df.columns.str.strip().str.lower()
    
    # Map target column dynamically from configuration fallbacks
    target = 'returned'
    if target not in df.columns:
        for potential_target in TARGET_COLUMN_FALLBACKS:
            if potential_target in df.columns:
                target = potential_target
                break
                
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not identified in dataset columns: {df.columns.tolist()}")
        
    # Feature Engineering: Derive is_weekend from order_weekday or order_datetime if available
    if 'order_weekday' in df.columns:
        df['is_weekend'] = df['order_weekday'].isin(['Saturday', 'Sunday']).astype(int)
    elif 'order_datetime' in df.columns:
        try:
            dt_series = pd.to_datetime(df['order_datetime'])
            df['order_weekday'] = dt_series.dt.day_name()
            df['order_hour'] = dt_series.dt.hour
            df['is_weekend'] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
        except Exception as e:
            print(f"Skipping datetime feature extraction: {e}")
            
    # Auto-detect features
    num_cols, cat_cols, ignored = detect_column_types(df, target)
    
    # Specific overrides if training on the standard cleaned dataset to keep things consistent
    # ensuring standard features are mapped correctly
    amazon_numerical = ['price', 'quantity', 'previous_returns_count', 'customer_total_orders', 
                        'discount_pct', 'review_rating', 'seller_rating', 'customer_tenure_days',
                        'is_prime_member', 'delivery_days', 'order_hour', 'is_weekend']
    amazon_categorical = ['product_category', 'payment_method', 'shipping_type', 'order_weekday']
    
    final_num = [c for c in amazon_numerical if c in df.columns]
    final_cat = [c for c in amazon_categorical if c in df.columns]
    
    # If standard features are not found, fall back to auto-detected features
    if len(final_num) == 0 and len(final_cat) == 0:
        final_num = num_cols
        final_cat = cat_cols
        
    print(f"[INFO] Numerical features: {final_num}")
    print(f"[INFO] Categorical features: {final_cat}")
    print(f"[INFO] Ignored features: {ignored}")
    
    # Train
    train_custom_model(df, target, final_num, final_cat)