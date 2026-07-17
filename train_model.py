import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Ensure the output models folder structure exists natively
os.makedirs("models", exist_ok=True)

# 1. Access the dataset path location
import sys

file_path = os.path.join("data", "amazon_returns_dataset_cleaned.xlsx")
if len(sys.argv) > 1:
    file_path = sys.argv[1]

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Missing base training data array at target location: '{file_path}'")

print(f"🔄 Ingesting transactional logs from: {file_path}")
if file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
else:
    df = pd.read_excel(file_path, engine="openpyxl")

# Standardize columns mapping
column_mapping = {
    'Order_ID': 'order_id', 'Order ID': 'order_id',
    'Product_ID': 'product_id', 'Product ID': 'product_id',
    'User_ID': 'customer_id', 'User ID': 'customer_id',
    'Product_Category': 'product_category', 'Product Category': 'product_category',
    'Product_Price': 'price', 'Product Price': 'price', 'Price': 'price',
    'Order_Quantity': 'quantity', 'Order Quantity': 'quantity', 'Quantity': 'quantity',
    'Return_Reason': 'return_reason', 'Return Reason': 'return_reason',
    'Return_Status': 'returned', 'Return Status': 'returned', 'returned': 'returned',
    'Days_to_Return': 'delivery_days', 'Days to Return': 'delivery_days',
    'User_Location': 'user_location', 'User Location': 'user_location',
    'Payment_Method': 'payment_method', 'Payment Method': 'payment_method',
    'Shipping_Method': 'shipping_type', 'Shipping Method': 'shipping_type', 'Shipping_Type': 'shipping_type', 'Shipping Type': 'shipping_type',
    'Discount_Applied': 'discount_pct', 'Discount Applied': 'discount_pct', 'Discount': 'discount_pct',
    'Order_Date': 'order_datetime', 'Order Date': 'order_datetime', 'order_date': 'order_datetime',
    'is_prime_member': 'is_prime_member', 'Prime_Member': 'is_prime_member',
    'previous_returns_count': 'previous_returns_count', 'previous returns count': 'previous_returns_count',
    'customer_total_orders': 'customer_total_orders', 'customer total orders': 'customer_total_orders',
    'customer_tenure_days': 'customer_tenure_days', 'customer tenure days': 'customer_tenure_days',
    'review_rating': 'review_rating', 'review rating': 'review_rating',
    'seller_rating': 'seller_rating', 'seller rating': 'seller_rating'
}

df = df.rename(columns=lambda x: column_mapping.get(x, x))
df = df.rename(columns=lambda x: column_mapping.get(x.strip(), x))
df.columns = df.columns.str.strip().str.lower()

# 2. Map target classification features dynamically
target = 'returned'
if target not in df.columns:
    # Fallback checks
    for potential_target in ['returned', 'return_status', 'returnstatus']:
        if potential_target in df.columns:
            target = potential_target
            break

if target not in df.columns:
    raise KeyError(f"Target column tracking vector '{target}' not identified in dataset schema columns: {df.columns.tolist()}")

# Convert string target outputs if text (e.g. 'Returned' vs 'Not Returned')
if df[target].dtype == object or df[target].dtype == str:
    print("Converting text target labels to binary classes...")
    df[target] = df[target].str.lower().str.strip().isin(['returned', '1', 'yes', 'true']).astype(int)

# Feature Engineering: Derive is_weekend from order_weekday or order_datetime
if 'order_weekday' in df.columns:
    df['is_weekend'] = df['order_weekday'].isin(['Saturday', 'Sunday']).astype(int)
elif 'order_datetime' in df.columns:
    try:
        dt_series = pd.to_datetime(df['order_datetime'])
        df['order_weekday'] = dt_series.dt.day_name()
        df['order_hour'] = dt_series.dt.hour
        df['is_weekend'] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
    except Exception as e:
        print(f"Skipping order_datetime parse: {e}")
        df['is_weekend'] = 0
else:
    df['is_weekend'] = 0

# Pool potential candidate attributes present across standard retail logs
numerical_candidates = [
    'price', 'quantity', 'previous_returns_count', 'customer_total_orders', 
    'discount_pct', 'review_rating', 'seller_rating', 'customer_tenure_days',
    'is_prime_member', 'delivery_days', 'order_hour', 'is_weekend'
]
categorical_candidates = [
    'product_category', 'payment_method', 'shipping_type', 'shipping_method', 
    'user_location', 'order_weekday'
]

numerical_features = [col for col in numerical_candidates if col in df.columns]
categorical_features = [col for col in categorical_candidates if col in df.columns]

print(f"✔ Active Quantitative Engine Features: {numerical_features}")
print(f"✔ Active Categorical Engine Features: {categorical_features}")

X = df[numerical_features + categorical_features]
y = df[target]

# Compute default/median values for imputation in app UI predictions
default_values = {}
for col in numerical_features:
    default_values[col] = float(df[col].median()) if col in df.columns else 0.0
for col in categorical_features:
    default_values[col] = str(df[col].mode()[0]) if (col in df.columns and not df[col].mode().empty) else 'Standard'

# 3. Formulate Preprocessing Transformers Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Synthesize End-to-End Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# 5. Segment Split Parameters (80% Fit Runway, 20% Evaluation Ground)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🏋️‍♂️ Processing tensor matrix distributions and fitting Random Forest weights...")
model_pipeline.fit(X_train, y_train)

# 6. Evaluate Inferences Metrics
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
print(f"\n📊 Model Training Complete! Baseline ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.2f}")
print("\n=== Classification Blueprint Profile ===")
print(classification_report(y_test, model_pipeline.predict(X_test)))

# 7. Compile Artifact Bundles alongside meta parameters for UI layout scaling
model_metadata = {
    'pipeline': model_pipeline,
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'default_values': default_values
}

joblib.dump(model_metadata, os.path.join('models', 'rto_predictor_model.pkl'))
print("💾 Success! Model architecture saved securely to 'models/rto_predictor_model.pkl'")