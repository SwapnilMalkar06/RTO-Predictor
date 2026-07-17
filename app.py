import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

st.set_page_config(page_title="Shopen Pulse - RTO Risk Engine", layout="wide")

# Custom premium CSS styling overrides
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font Overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Premium Header Card */
    .header-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-card h1 {
        color: #f8fafc !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.4rem !important;
        letter-spacing: -0.025em;
    }
    .header-card p {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0;
    }
    
    /* Styled widgets & containers */
    .section-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Custom styling for metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Styling for sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# Dynamically parse tracking attributes from baseline spreadsheets for drop-down option rendering
@st.cache_data
def load_base_metadata():
    file_path = os.path.join("data", "amazon_returns_dataset_cleaned.xlsx")
    if os.path.exists(file_path):
        return pd.read_excel(file_path, engine="openpyxl")
    return None

df_meta = load_base_metadata()

# Fetch serialized model pipelines
@st.cache_resource
def load_trained_pipeline():
    model_path = os.path.join('models', 'rto_predictor_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

meta_bundle = load_trained_pipeline()

# Header Layout Component
st.markdown("""
<div class="header-card">
    <h1>🛡️ Shopen Pulse — RTO Risk Control Center</h1>
    <p>Empowering digital merchants to inspect, predict, and intercept package return risk in real-time.</p>
</div>
""", unsafe_allow_html=True)

if meta_bundle is None:
    st.error("❌ Operational model artifact missing! Please execute `python train_model.py` in your terminal workspace to build your predictive file first.")
else:
    model_pipeline = meta_bundle['pipeline']
    numerical_features = meta_bundle['numerical_features']
    categorical_features = meta_bundle['categorical_features']
    
    # Setup workspace tab routing
    tab1, tab2, tab3 = st.tabs([
        "🎯 Single-Order Risk Profiler", 
        "📁 Bulk Transaction Scanner", 
        "📊 Analytics & Diagnostics"
    ])
    
    # ------------------ TAB 1: SINGLE-ORDER PROFILER ------------------
    with tab1:
        st.markdown("### 🎯 Live Transaction Analyzer")
        st.write("Submit individual transaction details below to evaluate RTO likelihood and generate operational recommendations.")
        
        col_form, col_result = st.columns([1.1, 0.9])
        
        with col_form:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📥 Transaction Entry Form")
            
            # Categories & selections loaded dynamically from historical spreadsheet if available
            categories = list(df_meta['product_category'].dropna().unique()) if df_meta is not None else ["Clothing", "Electronics", "Home", "Sports", "Toys", "Beauty", "Books"]
            shipping_types = list(df_meta['shipping_type'].dropna().unique()) if df_meta is not None else ["Standard", "Expedited", "Two-Day", "Same-Day"]
            payment_methods = list(df_meta['payment_method'].dropna().unique()) if df_meta is not None else ["Credit Card", "Debit Card", "COD", "UPI", "NetBanking", "Gift Card"]
            weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            # Form grid splits
            c1, c2 = st.columns(2)
            with c1:
                prod_cat = st.selectbox("Product Category", categories, index=0)
                price_val = st.number_input("Unit Price ($)", min_value=1.0, max_value=1000.0, value=50.0)
                qty_val = st.number_input("Quantity Purchased", min_value=1, max_value=50, value=1)
                ship_type = st.selectbox("Shipping Mode", shipping_types, index=0)
                deliv_days = st.slider("Expected Delivery (Days)", min_value=1, max_value=14, value=3)
                
            with c2:
                disc_val = st.slider("Discount Applied (%)", min_value=0, max_value=90, value=0)
                pay_method = st.selectbox("Payment Method", payment_methods, index=0)
                prev_returns = st.number_input("Previous Returns Count", min_value=0, max_value=50, value=0)
                total_orders = st.number_input("Customer Total Orders", min_value=1, max_value=500, value=5)
                tenure_days = st.slider("Customer Tenure (Days)", min_value=0, max_value=2000, value=180)
                
            st.markdown("#### ⚙️ Additional Attributes")
            c3, c4 = st.columns(2)
            with c3:
                is_prime = st.checkbox("Is Prime Member", value=False)
                ord_weekday = st.selectbox("Order Weekday", weekdays, index=0)
                ord_hour = st.slider("Order Hour (0-23)", min_value=0, max_value=23, value=12)
            with c4:
                prod_rating = st.slider("Product Review Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
                sel_rating = st.slider("Seller Rating", min_value=1.0, max_value=5.0, value=4.2, step=0.1)
                
            submit_trigger = st.button("Evaluate Order Risk Profile", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_result:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📊 Assessment Output")
            
            if submit_trigger:
                is_wknd = 1 if ord_weekday in ["Saturday", "Sunday"] else 0
                
                # Align variables exactly to training dataset format
                user_input_map = {
                    'product_category': prod_cat,
                    'price': price_val,
                    'quantity': qty_val,
                    'shipping_type': ship_type,
                    'delivery_days': deliv_days,
                    'discount_pct': disc_val,
                    'previous_returns_count': prev_returns,
                    'customer_total_orders': total_orders,
                    'customer_tenure_days': tenure_days,
                    'is_prime_member': 1 if is_prime else 0,
                    'order_weekday': ord_weekday,
                    'order_hour': ord_hour,
                    'is_weekend': is_wknd,
                    'review_rating': prod_rating,
                    'seller_rating': sel_rating,
                    'payment_method': pay_method
                }
                
                # Transform to DataFrame
                input_row = pd.DataFrame([user_input_map])
                
                with st.spinner("Analyzing operational RTO risk vectors..."):
                    time.sleep(0.4)
                    rto_score = model_pipeline.predict_proba(input_row)[0][1] * 100
                    
                st.session_state['last_rto_score'] = rto_score
                st.session_state['base_input'] = user_input_map
                
            if 'last_rto_score' in st.session_state:
                score = st.session_state['last_rto_score']
                base_data = st.session_state['base_input']
                
                # Display risk categorization banner
                if score < 35:
                    st.success(f"### ✅ Low RTO Risk: {score:.1f}%")
                    st.markdown("""
                    🟢 **Action Recommendation:** Safe transaction. Auto-forward to standard fulfillment queue and generate warehouse shipping labels.
                    """)
                elif score < 65:
                    st.warning(f"### ⚠️ Moderate RTO Risk: {score:.1f}%")
                    st.markdown("""
                    🟡 **Action Recommendation:** Verification requested. Fire address verification via WhatsApp notification to confirm client details.
                    """)
                else:
                    st.error(f"### 🚨 Critical High Risk: {score:.1f}%")
                    st.markdown("""
                    🔴 **Action Recommendation:** Shipping Hold. Suspend packaging operations. Order shows significant high-risk correlations. Resolve via manual telephone verification.
                    """)
                
                st.markdown("---")
                
                # Key Metrics Grid
                st.markdown("#### Primary Merchant Risk Drivers")
                m1, m2, m3 = st.columns(3)
                m1.metric("Payment Mode", str(base_data['payment_method']), "Risk Segment" if str(base_data['payment_method']) == "COD" else "Stable Channel")
                m2.metric("Return History", f"{base_data['previous_returns_count']} Ret.", "Prior return record" if base_data['previous_returns_count'] > 0 else "Normal")
                m3.metric("Promo Discount", f"{base_data['discount_pct']}% Off", "Impulse Factor" if base_data['discount_pct'] > 15 else "Standard")
                
                st.markdown("---")
                
                # What-if workspace subsegment
                st.markdown("#### 🛠️ What-If Scenario Simulator")
                st.write("Modify high-impact operational controls to see how changing business decisions adjusts risk predictions in real-time:")
                
                sim_delivery = st.slider("Adjust Delivery Speed (Days)", min_value=1, max_value=14, value=int(base_data['delivery_days']))
                sim_payment = st.selectbox("Adjust Payment Method", payment_methods, index=payment_methods.index(base_data['payment_method']))
                sim_discount = st.slider("Adjust Discount applied (%)", min_value=0, max_value=90, value=int(base_data['discount_pct']))
                
                simulated_map = base_data.copy()
                simulated_map['delivery_days'] = sim_delivery
                simulated_map['payment_method'] = sim_payment
                simulated_map['discount_pct'] = sim_discount
                
                simulated_row = pd.DataFrame([simulated_map])
                sim_score = model_pipeline.predict_proba(simulated_row)[0][1] * 100
                
                # Compare metrics
                delta = sim_score - score
                st.metric(
                    label="Adjusted RTO Risk Probability",
                    value=f"{sim_score:.1f}%",
                    delta=f"{delta:+.1f}%" if delta != 0 else "No Change",
                    delta_color="inverse"
                )
                
                if sim_score < score:
                    st.info("💡 **Decision Insight:** Lowering risk! Shorter expected delivery times, secure prepaid methods, or lower promotional discounts reduce returns.")
            else:
                st.info("💡 Please complete the transaction inputs and click 'Evaluate Order Risk Profile' to generate results.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    # ------------------ TAB 2: BULK TRANSACTION SCANNER ------------------
    with tab2:
        st.markdown("### 📁 Batch Order Risk Scan")
        st.write("Submit a spreadsheet of pending orders to scan for RTO risk anomalies before shipping.")
        
        # Download template help layout
        st.markdown("""
        > **Spreadsheet Column Format Instructions:**  
        > The upload file must include column headers matching:  
        > `price, quantity, previous_returns_count, customer_total_orders, discount_pct, review_rating, seller_rating, customer_tenure_days, is_prime_member, delivery_days, order_hour, order_weekday, product_category, payment_method, shipping_type`
        """)
        
        uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
        local_path = st.text_input("Or enter absolute path to local CSV/XLSX file on your computer:")
        
        file_to_process = None
        if uploaded_file is not None:
            file_to_process = uploaded_file
        elif local_path:
            if os.path.exists(local_path):
                file_to_process = local_path
            else:
                st.error("❌ Specified path does not exist on your computer. Please check the spelling and path formatting.")
                
        if file_to_process is not None:
            try:
                # Ingest dataset
                if isinstance(file_to_process, str):
                    if file_to_process.endswith('.csv'):
                        batch_df = pd.read_csv(file_to_process)
                    else:
                        batch_df = pd.read_excel(file_to_process, engine="openpyxl")
                else:
                    if file_to_process.name.endswith('.csv'):
                        batch_df = pd.read_csv(file_to_process)
                    else:
                        batch_df = pd.read_excel(file_to_process, engine="openpyxl")
                
                # Copy original columns for display
                display_df = batch_df.copy()
                
                # Standardize columns using a translation dictionary
                column_mapping = {
                    'order_id': 'order_id', 'order id': 'order_id',
                    'product_id': 'product_id', 'product id': 'product_id',
                    'user_id': 'customer_id', 'user id': 'customer_id', 'customer_id': 'customer_id',
                    'product_category': 'product_category', 'product category': 'product_category',
                    'product_price': 'price', 'product price': 'price', 'price': 'price',
                    'order_quantity': 'quantity', 'order quantity': 'quantity', 'quantity': 'quantity',
                    'return_reason': 'return_reason', 'return reason': 'return_reason',
                    'return_status': 'returned', 'return status': 'returned',
                    'days_to_return': 'delivery_days', 'days to return': 'delivery_days',
                    'user_location': 'user_location', 'user location': 'user_location',
                    'payment_method': 'payment_method', 'payment method': 'payment_method',
                    'shipping_method': 'shipping_type', 'shipping method': 'shipping_type', 'shipping_type': 'shipping_type', 'shipping type': 'shipping_type',
                    'discount_applied': 'discount_pct', 'discount applied': 'discount_pct', 'discount': 'discount_pct',
                    'order_date': 'order_datetime', 'order date': 'order_datetime', 'order_datetime': 'order_datetime',
                    'is_prime_member': 'is_prime_member', 'prime_member': 'is_prime_member', 'prime member': 'is_prime_member',
                    'previous_returns_count': 'previous_returns_count', 'previous returns count': 'previous_returns_count',
                    'customer_total_orders': 'customer_total_orders', 'customer total orders': 'customer_total_orders',
                    'customer_tenure_days': 'customer_tenure_days', 'customer tenure days': 'customer_tenure_days',
                    'review_rating': 'review_rating', 'review rating': 'review_rating',
                    'seller_rating': 'seller_rating', 'seller rating': 'seller_rating'
                }
                
                # Rename the dataframe columns for model scoring mapping
                scored_df = batch_df.rename(columns=lambda x: column_mapping.get(x, column_mapping.get(x.strip(), column_mapping.get(x.strip().lower(), x))))
                scored_df.columns = scored_df.columns.str.strip().str.lower()
                
                # Derivations
                if 'order_datetime' in scored_df.columns and 'order_weekday' not in scored_df.columns:
                    try:
                        dt_series = pd.to_datetime(scored_df['order_datetime'])
                        scored_df['order_weekday'] = dt_series.dt.day_name()
                        if 'order_hour' not in scored_df.columns:
                            scored_df['order_hour'] = dt_series.dt.hour
                    except Exception:
                        pass
                
                if 'order_weekday' in scored_df.columns and 'is_weekend' not in scored_df.columns:
                    scored_df['is_weekend'] = scored_df['order_weekday'].isin(['Saturday', 'Sunday']).astype(int)
                
                # Load default values from training metadata or define standard fallbacks
                meta_defaults = meta_bundle.get('default_values', {})
                standard_fallbacks = {
                    'price': 50.0, 'quantity': 1, 'previous_returns_count': 0, 'customer_total_orders': 5,
                    'discount_pct': 0, 'review_rating': 4.0, 'seller_rating': 4.0, 'customer_tenure_days': 180,
                    'is_prime_member': 0, 'delivery_days': 3, 'order_hour': 12, 'is_weekend': 0,
                    'product_category': 'Clothing', 'payment_method': 'Credit Card', 'shipping_type': 'Standard',
                    'order_weekday': 'Monday'
                }
                
                # Align columns to what pipeline expects
                features_ordered = numerical_features + categorical_features
                
                # Fill missing columns using default values
                for col in features_ordered:
                    if col not in scored_df.columns:
                        default_val = meta_defaults.get(col, standard_fallbacks.get(col, 0.0 if col in numerical_features else 'Standard'))
                        scored_df[col] = default_val
                
                # Extract inputs
                model_batch_in = scored_df[features_ordered]
                
                with st.spinner("Executing batch pipeline predictions..."):
                    rto_probs = model_pipeline.predict_proba(model_batch_in)[:, 1] * 100
                    display_df['rto_risk_probability (%)'] = np.round(rto_probs, 2)
                    
                    # Set risk categories
                    conditions = [
                        (display_df['rto_risk_probability (%)'] < 35),
                        (display_df['rto_risk_probability (%)'] >= 35) & (display_df['rto_risk_probability (%)'] < 65),
                        (display_df['rto_risk_probability (%)'] >= 65)
                    ]
                    categories_list = ['Low Risk', 'Moderate Risk', 'Critical High Risk']
                    display_df['rto_risk_category'] = np.select(conditions, categories_list, default='Low Risk')
                
                st.success("🎉 Batch Assessment Complete!")
                
                # Batch summary dashboard
                bm1, bm2, bm3 = st.columns(3)
                total_scanned = len(display_df)
                avg_scanned_risk = display_df['rto_risk_probability (%)'].mean()
                high_risk_scanned = (display_df['rto_risk_category'] == 'Critical High Risk').sum()
                high_risk_scanned_pct = (high_risk_scanned / total_scanned) * 100
                
                bm1.metric("Total Scanned Orders", f"{total_scanned}")
                bm2.metric("Average RTO Risk Score", f"{avg_scanned_risk:.1f}%")
                bm3.metric("High-Risk Suspects", f"{high_risk_scanned} ({high_risk_scanned_pct:.1f}%)")
                
                st.markdown("#### 📋 Assessment Output Queue")
                # Put critical prediction outputs at front
                col_order = ['rto_risk_category', 'rto_risk_probability (%)']
                other_cols = [c for c in display_df.columns if c not in col_order]
                st.dataframe(display_df[col_order + other_cols], use_container_width=True)
                
                # Export options
                csv_export = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Processed Predictions Log (CSV)",
                    data=csv_export,
                    file_name="shopen_rto_batch_predictions.csv",
                    mime="text/csv",
                    type="primary"
                )
            except Exception as e:
                st.error(f"❌ Error parsing file structure: {e}")
                
    # ------------------ TAB 3: ANALYTICS & DIAGNOSTICS ------------------
    with tab3:
        st.markdown("### 📊 Historical Return Reason Diagnostics")
        st.write("Visual diagnostics dashboard powered by historical transaction registries.")
        
        if df_meta is None:
            st.warning("⚠️ Historical analytics spreadsheet database missing or unreadable.")
        else:
            # Main high-level statistics cards
            tot_logs = len(df_meta)
            global_rto = df_meta['returned'].mean() * 100
            clothing_only = df_meta[df_meta['product_category'] == 'Clothing']
            clothing_rto_rate = clothing_only['returned'].mean() * 100
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Historical Logs Ingested", f"{tot_logs} orders")
            k2.metric("Base Return Rate", f"{global_rto:.1f}%")
            k3.metric("Apparel Category Return Rate", f"{clothing_rto_rate:.1f}%", delta="Critical High RTO")
            
            st.markdown("---")
            
            # Row 1 charts
            row1_c1, row1_c2 = st.columns(2)
            
            with row1_c1:
                st.markdown("#### 👚 Return Rates by Product Category")
                cat_data = df_meta.groupby('product_category')['returned'].mean().reset_index()
                cat_data['RTO Rate (%)'] = np.round(cat_data['returned'] * 100, 2)
                cat_data = cat_data.sort_values(by='RTO Rate (%)', ascending=False)
                st.bar_chart(cat_data, x='product_category', y='RTO Rate (%)')
                st.caption("Clothing represents a massive outlier, exhibiting a near-total return probability.")
                
            with row1_c2:
                st.markdown("#### 🔍 Primary Reasons for Customer Returns")
                ret_logs = df_meta[df_meta['returned'] == 1]
                if 'return_reason' in ret_logs.columns:
                    reasons_count = ret_logs['return_reason'].value_counts().reset_index()
                    reasons_count.columns = ['Return Reason', 'Number of Returns']
                    st.bar_chart(reasons_count, x='Return Reason', y='Number of Returns')
                    st.caption("Size/fit errors and shipping errors (Wrong item) represent the leading contributors.")
                else:
                    st.info("Missing return reason labels in database logs.")
                    
            st.markdown("---")
            
            # Row 2 charts
            row2_c1, row2_c2 = st.columns(2)
            
            with row2_c1:
                st.markdown("#### 💳 Return Rate by Payment Method")
                pay_data = df_meta.groupby('payment_method')['returned'].mean().reset_index()
                pay_data['RTO Rate (%)'] = np.round(pay_data['returned'] * 100, 2)
                pay_data = pay_data.sort_values(by='RTO Rate (%)', ascending=False)
                st.bar_chart(pay_data, x='payment_method', y='RTO Rate (%)')
                st.caption("Cash on Delivery (COD) and standard channels show balanced ratios.")
                
            with row2_c2:
                st.markdown("#### 👑 Prime Membership Return Impact")
                prime_data = df_meta.groupby('is_prime_member')['returned'].mean().reset_index()
                prime_data['Segment'] = prime_data['is_prime_member'].map({0: 'Non-Prime Member', 1: 'Prime Member'})
                prime_data['RTO Rate (%)'] = np.round(prime_data['returned'] * 100, 2)
                st.bar_chart(prime_data, x='Segment', y='RTO Rate (%)')
                st.caption("Non-Prime customers show a ~9% higher return likelihood than Prime members.")