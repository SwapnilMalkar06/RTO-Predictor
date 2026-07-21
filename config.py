"""
Centralized Configuration Module for the Universal Risk Predictor.
Defines files paths, dirty-to-standard column mappings, and column auto-detection parameters.
"""

import os

# --- Path Configurations ---
DEFAULT_DATASET_PATH = os.path.join("data", "amazon_returns_dataset_cleaned.xlsx")
MODEL_OUTPUT_PATH = os.path.join("models", "rto_predictor_model.pkl")

# --- Column Detection & Standardization Mappings ---
TARGET_COLUMN_FALLBACKS = [
    "returned",
    "return_status",
    "rto",
    "returned_status",
    "status",
    "returnstatus"
]

COLUMN_MAPPING = {
    # Keys
    "Order_ID": "order_id",
    "Order ID": "order_id",
    "Product_ID": "product_id",
    "Product ID": "product_id",
    "User_ID": "customer_id",
    "User ID": "customer_id",
    # Features
    "Product_Category": "product_category",
    "Product Category": "product_category",
    "Product_Price": "price",
    "Product Price": "price",
    "Price": "price",
    "Order_Quantity": "quantity",
    "Order Quantity": "quantity",
    "Quantity": "quantity",
    "Return_Reason": "return_reason",
    "Return Reason": "return_reason",
    "Return_Status": "returned",
    "Return Status": "returned",
    "returned": "returned",
    "Days_to_Return": "delivery_days",
    "Days to Return": "delivery_days",
    "User_Location": "user_location",
    "User Location": "user_location",
    "Payment_Method": "payment_method",
    "Payment Method": "payment_method",
    "Shipping_Method": "shipping_type",
    "Shipping Method": "shipping_type",
    "Shipping_Type": "shipping_type",
    "Shipping Type": "shipping_type",
    "Discount_Applied": "discount_pct",
    "Discount Applied": "discount_pct",
    "Discount": "discount_pct",
    "Order_Date": "order_datetime",
    "Order Date": "order_datetime",
    "order_date": "order_datetime",
    "is_prime_member": "is_prime_member",
    "Prime_Member": "is_prime_member",
    "previous_returns_count": "previous_returns_count",
    "previous returns count": "previous_returns_count",
    "customer_total_orders": "customer_total_orders",
    "customer total orders": "customer_total_orders",
    "customer_tenure_days": "customer_tenure_days",
    "customer tenure days": "customer_tenure_days",
    "review_rating": "review_rating",
    "review rating": "review_rating",
    "seller_rating": "seller_rating",
    "seller rating": "seller_rating"
}
