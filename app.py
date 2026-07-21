from typing import Any, Optional, Dict, List
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from train_model import train_custom_model, detect_column_types
from config import (
    DEFAULT_DATASET_PATH,
    MODEL_OUTPUT_PATH,
    COLUMN_MAPPING,
    TARGET_COLUMN_FALLBACKS
)

st.set_page_config(
    page_title="Universal Risk Predictor - Dynamic RTO Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Font overrides */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card Component */
    .header-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-card h1 {
        color: #f8fafc !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.025em;
    }
    .header-card p {
        font-size: 1.05rem;
        color: #94a3b8;
        margin: 0;
    }
    
    /* Segment Panels */
    .section-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Metric Card styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Status indicators */
    .success-alert {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        color: #14532d;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .warning-alert {
        background-color: #fffbef;
        border-left: 5px solid #d97706;
        color: #78350f;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .error-alert {
        background-color: #fef2f2;
        border-left: 5px solid #dc2626;
        color: #7f1d1d;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Premium Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_default_model() -> Optional[Dict[str, Any]]:
    """
    Loads the default serialized model metadata bundle from disk.
    Does not render UI sidebar errors inside caching layer to prevent side-effects.
    """
    if os.path.exists(MODEL_OUTPUT_PATH):
        return joblib.load(MODEL_OUTPUT_PATH)
    return None

@st.cache_data
def load_default_dataset() -> Optional[pd.DataFrame]:
    """
    Loads the default Amazon returns cleaned baseline dataset.
    Standardizes schema headers using Column Mapping configuration.
    """
    if os.path.exists(DEFAULT_DATASET_PATH):
        df = pd.read_excel(DEFAULT_DATASET_PATH, engine="openpyxl")
        df = df.rename(columns=lambda x: COLUMN_MAPPING.get(x, x))
        df = df.rename(columns=lambda x: COLUMN_MAPPING.get(x.strip(), x))
        df.columns = df.columns.str.strip().str.lower()
        
        # Derive weekend mapping if not present
        if 'order_weekday' in df.columns:
            df['is_weekend'] = df['order_weekday'].isin(['Saturday', 'Sunday']).astype(int)
        elif 'order_datetime' in df.columns:
            try:
                dt_series = pd.to_datetime(df['order_datetime'])
                df['order_weekday'] = dt_series.dt.day_name()
                df['order_hour'] = dt_series.dt.hour
                df['is_weekend'] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            except Exception:
                df['is_weekend'] = 0
        else:
            df['is_weekend'] = 0
            
        return df
    return None

def get_clean_aggregated_importances(active_model: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Standardizes and aggregates feature importance weights from categorical/OHE columns.
    """
    if active_model is None:
        return {}
    agg = active_model.get('aggregated_importances', {})
    # If agg is empty or all values are 0.0, try to reconstruct it from detailed importances
    if len(agg) == 0 or sum(agg.values()) == 0.0:
        detailed = active_model.get('feature_importances', {})
        num_feats = active_model.get('numerical_features', [])
        cat_feats = active_model.get('categorical_features', [])
        
        agg = {col: 0.0 for col in num_feats + cat_feats}
        for k, v in detailed.items():
            clean_k = k
            if k.startswith("num__"):
                clean_k = k[5:]
            elif k.startswith("cat__"):
                clean_k = k[5:]
                
            for col in num_feats:
                if clean_k == col:
                    agg[col] += v
                    break
            for col in cat_feats:
                if clean_k.startswith(col + "_") or clean_k == col:
                    agg[col] += v
                    break
                    
        # Standardize sum to 1.0
        s = sum(agg.values())
        if s > 0:
            agg = {k: v / s for k, v in agg.items()}
    return agg

# --- State Management Initialization ---
if 'active_model' not in st.session_state:
    try:
        st.session_state['active_model'] = load_default_model()
    except Exception as e:
        st.sidebar.error(f"Error loading default model: {e}")
        st.session_state['active_model'] = None
    
if 'active_df' not in st.session_state:
    try:
        st.session_state['active_df'] = load_default_dataset()
    except Exception as e:
        st.sidebar.error(f"Error loading default dataset: {e}")
        st.session_state['active_df'] = None

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9598/9598096.png", width=60)
    st.markdown("### **Risk Predictor Console**")
    st.write("Dynamic RTO & returns control cockpit.")
    st.markdown("---")
    
    # Model Status Card
    if st.session_state['active_model'] is not None:
        model_meta = st.session_state['active_model']
        target_name = model_meta.get('target_column', 'returned')
        acc = model_meta.get('metrics', {}).get('accuracy', 0.85)
        
        st.success("🟢 **Operational Model Loaded**")
        st.markdown(f"**Target Col:** `{target_name}`")
        st.markdown(f"**Num Features:** `{len(model_meta['numerical_features'])}`")
        st.markdown(f"**Cat Features:** `{len(model_meta['categorical_features'])}`")
        st.markdown(f"**Accuracy:** `{acc*100:.1f}%`")
    else:
        st.error("🔴 **No Model Loaded**")
        st.info("Train a model in the 'Model Studio' tab to enable predictions.")
        
    st.markdown("---")
    # Quick reset button
    if st.button("Reset to Default Settings"):
        try:
            st.session_state['active_model'] = load_default_model()
        except Exception as e:
            st.sidebar.error(f"Error resetting default model: {e}")
            st.session_state['active_model'] = None
        try:
            st.session_state['active_df'] = load_default_dataset()
        except Exception as e:
            st.sidebar.error(f"Error resetting default dataset: {e}")
            st.session_state['active_df'] = None
        if 'scored_bulk_df' in st.session_state:
            st.session_state['scored_bulk_df'] = None
        if 'bulk_filename' in st.session_state:
            st.session_state['bulk_filename'] = None
        st.rerun()

# --- Main Layout Header ---
st.markdown("""
<div class="header-card">
    <h1>🛡️ Universal Risk & RTO Predictive Suite</h1>
    <p>Upload custom datasets, dynamically train classification models, run batch files, and analyze returns diagnostics.</p>
</div>
""", unsafe_allow_html=True)

# Define application tabs
tab_studio, tab_profiler, tab_bulk, tab_report = st.tabs([
    "⚙️ Model Studio (Train & Load)",
    "🎯 Dynamic Risk Profiler",
    "📁 Bulk Transaction Scanner",
    "📊 Interactive Graphical Report"
])

# ==========================================
# TAB: MODEL STUDIO (Train & Load)
# ==========================================
with tab_studio:
    st.markdown("### ⚙️ Machine Learning Model Studio")
    st.write("Upload a raw dataset, dynamically define the schema, train the Random Forest pipeline, and review metrics.")
    
    col_input, col_metrics = st.columns([1.1, 0.9])
    
    with col_input:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📥 Dataset Ingestion")
        
        dataset_source = st.radio(
            "Select Training Dataset Source:",
            ["Use Default Amazon Returns Cleaned Dataset", "Upload Custom CSV or Excel File"]
        )
        
        raw_df = None
        if dataset_source == "Use Default Amazon Returns Cleaned Dataset":
            raw_df = load_default_dataset()
            if raw_df is not None:
                st.info("Loaded default Amazon returns dataset successfully.")
            else:
                st.error("Missing default dataset. Please upload a file manually.")
        else:
            uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        raw_df = pd.read_csv(uploaded_file)
                    else:
                        raw_df = pd.read_excel(uploaded_file, engine="openpyxl")
                    st.success(f"Successfully uploaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error parsing file: {e}")
                    
        if raw_df is not None:
            # Clean dataframe column names
            raw_df.columns = raw_df.columns.str.strip()
            
            # Brief preview of data
            st.markdown("#### 🔍 Dataset Summary")
            st.write(f"**Dimensions:** {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
            
            # Select target variable
            cols_list = sorted(list(raw_df.columns))
            # Auto-detect target column index using central configuration fallbacks
            target_default_idx = 0
            for idx, c in enumerate(cols_list):
                if c.lower() in [f.lower() for f in TARGET_COLUMN_FALLBACKS]:
                    target_default_idx = idx
                    break
                    
            target_col = st.selectbox("Select Target Class Column:", cols_list, index=target_default_idx)
            
            # Dynamic Feature Mapping
            num_cols, cat_cols, ignored = detect_column_types(raw_df, target_col)
            
            st.markdown("#### ⚡ Dynamic Feature Selection")
            st.write("Exclude columns or adjust feature types inferred by the model:")
            
            all_possible_features = [c for c in raw_df.columns if c != target_col]
            
            selected_features = st.multiselect(
                "Features to include in model pipeline:",
                all_possible_features,
                default=[c for c in all_possible_features if c not in ignored]
            )
            
            # Override feature type configuration
            st.markdown("##### Categorize Feature Columns:")
            feat_types = {}
            col_feat_1, col_feat_2 = st.columns(2)
            
            for idx, feat in enumerate(selected_features):
                # Put in alternate columns
                target_col_grid = col_feat_1 if idx % 2 == 0 else col_feat_2
                with target_col_grid:
                    inferred_type = "Numerical" if feat in num_cols else "Categorical"
                    feat_types[feat] = st.selectbox(
                        f"Type for `{feat}`:",
                        ["Numerical", "Categorical"],
                        index=0 if inferred_type == "Numerical" else 1,
                        key=f"feat_type_{feat}"
                    )
                    
            # Separate features based on user settings
            final_num_features = [f for f, t in feat_types.items() if t == "Numerical"]
            final_cat_features = [f for f, t in feat_types.items() if t == "Categorical"]
            
            # Training action button
            if st.button("🚀 Train Machine Learning Pipeline", type="primary", use_container_width=True):
                if len(selected_features) == 0:
                    st.error("Please select at least 1 feature column to train the model.")
                else:
                    with st.spinner("Executing pipeline preprocessing and RF training..."):
                        try:
                            # Standardize column casing/naming to lowercase for model compatibility
                            df_train = raw_df.copy()
                            df_train.columns = df_train.columns.str.strip().str.lower()
                            
                            lower_target = target_col.lower()
                            lower_num = [c.lower() for c in final_num_features]
                            lower_cat = [c.lower() for c in final_cat_features]
                            
                            # Clean target values to binary integers
                            trained_meta = train_custom_model(df_train, lower_target, lower_num, lower_cat)
                            
                            st.session_state['active_model'] = trained_meta
                            st.session_state['active_df'] = df_train
                            st.success("🎉 ML pipeline trained and loaded as operational model!")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Training failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_metrics:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📊 Active Model Blueprint")
        
        if st.session_state['active_model'] is not None:
            active = st.session_state['active_model']
            metrics = active['metrics']
            
            # Metric grid
            st.metric("Accuracy Score", f"{metrics['accuracy']*100:.2f}%")
            
            # Show Classification Report
            st.markdown("#### Classification Report Details")
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df.style.format(precision=3), use_container_width=True)
            
            # Plot Feature Importance using Plotly
            st.markdown("#### Feature Importance Profile")
            agg_importances = get_clean_aggregated_importances(active)
            if len(agg_importances) > 0:
                imp_df = pd.DataFrame({
                    'Feature': list(agg_importances.keys()),
                    'Importance': list(agg_importances.values())
                }).sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(
                    imp_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Aggregated Feature Importance',
                    color='Importance',
                    color_continuous_scale='blues',
                    template='plotly_white'
                )
                fig_imp.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance data not found in model.")
        else:
            st.info("Load or train a model to view metrics.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB: DYNAMIC RISK PROFILER
# ==========================================
with tab_profiler:
    st.markdown("### 🎯 Single-Order Risk Profiler")
    st.write("Manually compile dynamic forms to predict return risks for single transactions in real-time.")
    
    if st.session_state['active_model'] is None:
        st.warning("⚠️ Operational model artifact missing! Please configure and train a model in the 'Model Studio' tab first.")
    else:
        model_bundle = st.session_state['active_model']
        model_pipeline = model_bundle['pipeline']
        numerical_features = model_bundle['numerical_features']
        categorical_features = model_bundle['categorical_features']
        feature_metadata = model_bundle['feature_metadata']
        
        col_form, col_eval = st.columns([1.1, 0.9])
        
        with col_form:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📥 Transaction Inputs Form")
            
            user_inputs = {}
            
            # Grid display inputs (split numerical and categorical)
            c1, c2 = st.columns(2)
            
            # Process numerical inputs
            for idx, num_feat in enumerate(numerical_features):
                target_column = c1 if idx % 2 == 0 else c2
                with target_column:
                    meta = feature_metadata[num_feat]
                    min_val = float(meta['min'])
                    max_val = float(meta['max'])
                    median_val = float(meta['median'])
                    
                    # Tweak format based on integer values
                    if min_val.is_integer() and max_val.is_integer():
                        # Render dynamic slider
                        user_inputs[num_feat] = st.slider(
                            f"{num_feat.replace('_', ' ').title()}",
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(median_val),
                            step=1
                        )
                    else:
                        # Render number input
                        user_inputs[num_feat] = st.number_input(
                            f"{num_feat.replace('_', ' ').title()}",
                            min_value=min_val,
                            max_value=max_val,
                            value=median_val,
                            format="%.2f"
                        )
                        
            # Process categorical inputs
            for idx, cat_feat in enumerate(categorical_features):
                target_column = c1 if (idx + len(numerical_features)) % 2 == 0 else c2
                with target_column:
                    meta = feature_metadata[cat_feat]
                    unique_vals = meta['unique_values']
                    mode_val = meta['mode']
                    
                    # Ensure mode is inside unique vals list
                    default_idx = unique_vals.index(mode_val) if mode_val in unique_vals else 0
                    
                    user_inputs[cat_feat] = st.selectbox(
                        f"{cat_feat.replace('_', ' ').title()}",
                        unique_vals,
                        index=default_idx
                    )
                    
            submit_eval = st.button("Evaluate Transaction Risk Profile", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_eval:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📊 Assessment Output")
            
            if submit_eval:
                # Convert inputs to DataFrame
                input_row = pd.DataFrame([user_inputs])
                
                with st.spinner("Analyzing operational RTO risk vectors..."):
                    time.sleep(0.3)
                    prob_scores = model_pipeline.predict_proba(input_row)[0]
                    rto_prob = prob_scores[1] * 100
                    
                st.session_state['eval_rto_score'] = rto_prob
                st.session_state['eval_inputs'] = user_inputs
                
            if 'eval_rto_score' in st.session_state:
                score = st.session_state['eval_rto_score']
                base_inputs = st.session_state['eval_inputs']
                
                # Show premium risk category banner
                if score < 35:
                    st.markdown(f"""
                    <div class="success-alert">
                        <h3>✅ Low Return Risk: {score:.1f}%</h3>
                        <p><strong>Action Recommendation:</strong> Safe transaction. Process order normally and generate fulfillment labels.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif score < 65:
                    st.markdown(f"""
                    <div class="warning-alert">
                        <h3>⚠️ Moderate Return Risk: {score:.1f}%</h3>
                        <p><strong>Action Recommendation:</strong> Verification requested. Recommend dispatching automated order confirmation notification via SMS or WhatsApp.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="error-alert">
                        <h3>🚨 Critical High Risk: {score:.1f}%</h3>
                        <p><strong>Action Recommendation:</strong> Hold Order. Flagged for fraud or high likelihood of return. Require manual customer verification call before shipment.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # What-If Scenario Simulator dynamically built based on model feature importances
                st.markdown("---")
                st.markdown("#### 🛠️ Dynamic What-If Simulator")
                st.write("Modify key numeric drivers below to evaluate real-time adjustments on return likelihood:")
                
                # Find top 2 numerical features based on aggregate importance
                agg_imp = get_clean_aggregated_importances(model_bundle)
                num_imps = {k: v for k, v in agg_imp.items() if k in numerical_features}
                sorted_num_feats = sorted(num_imps.keys(), key=lambda x: num_imps[x], reverse=True)
                
                top_feats = sorted_num_feats[:2] if len(sorted_num_feats) >= 2 else sorted_num_feats
                
                sim_inputs = base_inputs.copy()
                
                # Render simulator sliders
                for feat in top_feats:
                    meta = feature_metadata[feat]
                    min_v = float(meta['min'])
                    max_v = float(meta['max'])
                    current_v = float(base_inputs[feat])
                    
                    if min_v.is_integer() and max_v.is_integer():
                        sim_inputs[feat] = st.slider(
                            f"Simulate `{feat.replace('_', ' ').title()}`:",
                            min_value=int(min_v),
                            max_value=int(max_v),
                            value=int(current_v),
                            step=1,
                            key=f"sim_{feat}"
                        )
                    else:
                        sim_inputs[feat] = st.slider(
                            f"Simulate `{feat.replace('_', ' ').title()}`:",
                            min_value=min_v,
                            max_value=max_v,
                            value=current_v,
                            key=f"sim_{feat}"
                        )
                        
                # Predict simulated row
                sim_row = pd.DataFrame([sim_inputs])
                sim_score = model_pipeline.predict_proba(sim_row)[0][1] * 100
                
                # Compare metrics
                delta = sim_score - score
                st.metric(
                    label="Adjusted Risk Probability",
                    value=f"{sim_score:.1f}%",
                    delta=f"{delta:+.1f}%" if delta != 0 else "No Change",
                    delta_color="inverse"
                )
            else:
                st.info("💡 Input values and click 'Evaluate Transaction Risk Profile' to examine predictions.")
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB: BULK TRANSACTION SCANNER
# ==========================================
with tab_bulk:
    st.markdown("### 📁 Batch Order Risk Scanner")
    st.write("Scan entire order registries to categorize returns risk before warehouse loading.")
    
    if st.session_state['active_model'] is None:
        st.warning("⚠️ Operational model artifact missing! Please configure and train a model in the 'Model Studio' tab first.")
    else:
        model_bundle = st.session_state['active_model']
        model_pipeline = model_bundle['pipeline']
        numerical_features = model_bundle['numerical_features']
        categorical_features = model_bundle['categorical_features']
        feature_metadata = model_bundle['feature_metadata']
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📥 Upload Pending Registry")
        st.write("Upload a CSV/Excel file to score risk predictions across all rows.")
        
        # Display required features
        st.markdown(f"**Required Features for Scoring:** `{', '.join(numerical_features + categorical_features)}`")
        
        bulk_file = st.file_uploader("Upload pending file:", type=["csv", "xlsx"], key="bulk_uploader")
        
        if bulk_file is not None:
            try:
                # Load
                if bulk_file.name.endswith('.csv'):
                    bulk_df = pd.read_csv(bulk_file)
                else:
                    bulk_df = pd.read_excel(bulk_file, engine="openpyxl")
                    
                # Clean column headers
                original_cols = list(bulk_df.columns)
                bulk_df.columns = bulk_df.columns.str.strip().str.lower()
                
                # Verify and align columns
                missing_features = []
                aligned_df = pd.DataFrame(index=bulk_df.index)
                
                for num_col in numerical_features:
                    if num_col in bulk_df.columns:
                        aligned_df[num_col] = bulk_df[num_col]
                    else:
                        missing_features.append(num_col)
                        aligned_df[num_col] = feature_metadata[num_col]['median']
                        
                for cat_col in categorical_features:
                    if cat_col in bulk_df.columns:
                        aligned_df[cat_col] = bulk_df[cat_col]
                    else:
                        missing_features.append(cat_col)
                        aligned_df[cat_col] = feature_metadata[cat_col]['mode']
                        
                # Warn if missing features were filled with defaults
                if len(missing_features) > 0:
                    st.warning(f"⚠️ **Note:** The following columns were missing and imputed using training metadata defaults: `{', '.join(missing_features)}`")
                    
                # Run prediction
                with st.spinner("Scoring batch items..."):
                    probs = model_pipeline.predict_proba(aligned_df)[:, 1] * 100
                    
                # Format final results
                results_df = bulk_df.copy()
                results_df['risk_probability (%)'] = np.round(probs, 2)
                
                conditions = [
                    (results_df['risk_probability (%)'] < 35),
                    (results_df['risk_probability (%)'] >= 35) & (results_df['risk_probability (%)'] < 65),
                    (results_df['risk_probability (%)'] >= 65)
                ]
                results_df['risk_category'] = np.select(conditions, ['Low Risk', 'Moderate Risk', 'Critical High Risk'], default='Low Risk')
                
                # Save scored bulk dataset to session state for visualization in Graphical Report
                st.session_state['scored_bulk_df'] = results_df
                st.session_state['bulk_filename'] = bulk_file.name
                
                st.success("🎉 Batch scoring analysis complete!")
                
                # Show summary widgets
                s_col1, s_col2, s_col3 = st.columns(3)
                total_scored = len(results_df)
                avg_prob = results_df['risk_probability (%)'].mean()
                high_risk_count = (results_df['risk_category'] == 'Critical High Risk').sum()
                
                s_col1.metric("Total Records Scanned", f"{total_scored}")
                s_col2.metric("Mean Return Risk Score", f"{avg_prob:.1f}%")
                s_col3.metric("Critical High-Risk Suspects", f"{high_risk_count} ({ (high_risk_count/total_scored)*100 if total_scored > 0 else 0:.1f}%)")
                
                # Split display: left table, right distribution pie
                res_col1, res_col2 = st.columns([1.2, 0.8])
                with res_col1:
                    st.markdown("#### Scored Results Preview")
                    show_cols = ['risk_category', 'risk_probability (%)'] + [c for c in results_df.columns if c not in ['risk_category', 'risk_probability (%)']]
                    # Limit rendering preview to 1,000 rows to prevent WebSocket MessageSizeError on large uploads
                    st.dataframe(results_df[show_cols].head(1000), use_container_width=True)
                    if len(results_df) > 1000:
                        st.caption(f"⚠️ Showing first 1,000 rows of {len(results_df):,} total records. Use the download button below to retrieve the full dataset.")
                    
                    csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Full Predictions Spreadsheet (CSV)",
                        data=csv_bytes,
                        file_name="batch_risk_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
                with res_col2:
                    st.markdown("#### Risk Distribution")
                    cat_counts = results_df['risk_category'].value_counts().reset_index()
                    cat_counts.columns = ['Risk Segment', 'Count']
                    
                    fig_pie = px.pie(
                        cat_counts,
                        values='Count',
                        names='Risk Segment',
                        color='Risk Segment',
                        color_discrete_map={
                            'Low Risk': '#16a34a',
                            'Moderate Risk': '#eab308',
                            'Critical High Risk': '#dc2626'
                        },
                        title="Distribution of Scored Records",
                        hole=0.4
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error executing batch scoring: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB: INTERACTIVE GRAPHICAL REPORT
# ==========================================
with tab_report:
    st.markdown("### 📊 Interactive Returns Diagnostic Studio")
    st.write("Create real-time filtered, interactive graphical reports across any dataset configured in Model Studio.")
    
    # Determine which dataset to use for graphical reporting
    if 'scored_bulk_df' in st.session_state and st.session_state['scored_bulk_df'] is not None:
        df_report = st.session_state['scored_bulk_df']
        source_name = f"uploaded Bulk Scanner file (`{st.session_state.get('bulk_filename', 'bulk_file')}`)"
        is_bulk_active = True
    else:
        df_report = st.session_state.get('active_df', None)
        source_name = "active database dataset"
        is_bulk_active = False
        
    if df_report is None:
        st.warning("⚠️ Active dataset missing! Please load/train a model in Model Studio or upload a file in Bulk Transaction Scanner first.")
    else:
        # Display source banner
        if is_bulk_active:
            st.info(f"📊 **Data Source:** Currently visualizing predictions from the {source_name}. To switch back to the database dataset, click 'Reset to Default Settings' in the sidebar.")
        else:
            st.info(f"📊 **Data Source:** Visualizing the {source_name} from the data folder.")
            
        # Determine target column
        active_model = st.session_state['active_model']
        target_col = active_model.get('target_column', 'returned') if active_model is not None else 'returned'
        
        # Fallback check
        if target_col not in df_report.columns:
            if 'risk_category' in df_report.columns:
                target_col = 'risk_category'
            else:
                for col in df_report.columns:
                    if col.lower() in [f.lower() for f in TARGET_COLUMN_FALLBACKS]:
                        target_col = col
                        break
                        
        # Verify the target column is indeed binary or categorical with low cardinality
        target_is_binary = False
        if target_col in df_report.columns:
            unique_targets = df_report[target_col].dropna().unique()
            if len(unique_targets) <= 3:
                target_is_binary = True
                
        # --- Live Filtering Interface ---
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🔍 Real-time Dataset Filters")
        st.write("Select filters to refine the data used to generate the charts dynamically:")
        
        # Columns available for filtering
        filter_candidates = []
        for c in df_report.columns:
            if c == target_col:
                continue
            # Exclude ID and high-cardinality string columns
            if c.lower() in ['order_id', 'customer_id', 'product_id', 'id', 'user_id', 'order_datetime']:
                continue
            if not pd.api.types.is_numeric_dtype(df_report[c]) and df_report[c].nunique() > 50:
                continue
            filter_candidates.append(c)
            
        selected_filter_cols = st.multiselect("Select columns to filter by:", filter_candidates, default=filter_candidates[:2] if len(filter_candidates) >= 2 else filter_candidates)
        
        filtered_df = df_report.copy()
        
        if len(selected_filter_cols) > 0:
            filt_cols = st.columns(min(len(selected_filter_cols), 4))
            for idx, col in enumerate(selected_filter_cols):
                col_ui = filt_cols[idx % 4]
                with col_ui:
                    if pd.api.types.is_numeric_dtype(df_report[col]):
                        # Numerical filter
                        min_val = float(df_report[col].min())
                        max_val = float(df_report[col].max())
                        
                        # Handle case where min_val equals max_val
                        if min_val == max_val:
                            st.write(f"`{col}` constant = {min_val}")
                        else:
                            range_vals = st.slider(
                                f"Range for `{col}`:",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"filt_slider_{col}"
                            )
                            filtered_df = filtered_df[(filtered_df[col] >= range_vals[0]) & (filtered_df[col] <= range_vals[1])]
                    else:
                        # Categorical filter
                        options = sorted([str(x) for x in df_report[col].dropna().unique()])
                        selected_vals = st.multiselect(
                            f"Filter `{col}`:",
                            options,
                            default=options,
                            key=f"filt_select_{col}"
                        )
                        if len(selected_vals) > 0:
                            filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_vals)]
                            
            pct = (len(filtered_df) / len(df_report)) * 100 if len(df_report) > 0 else 0.0
            st.info(f"⚡ **Active Filters:** Showing {len(filtered_df)} of {len(df_report)} records ({pct:.1f}%)")
        else:
            st.info("No active filters. Showing 100% of the dataset.")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # If dataset is empty after filtering
        if len(filtered_df) == 0:
            st.error("❌ The selected filters return 0 records. Adjust the filters to display graphics.")
        else:
            # --- Graphical Diagnostic Dashboard Grid ---
            row1_1, row1_2 = st.columns(2)
            
            with row1_1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("🎯 Target Variable Distribution")
                
                if target_col in filtered_df.columns:
                    unique_count = filtered_df[target_col].nunique()
                    
                    if unique_count <= 10:
                        target_counts = filtered_df[target_col].value_counts().reset_index()
                        target_counts.columns = [target_col, 'Count']
                        
                        # Humanize target mapping for binary classification columns
                        if target_is_binary:
                            unique_vals = set(target_counts[target_col])
                            if unique_vals.issubset({0, 1}):
                                target_counts['Label'] = target_counts[target_col].map({0: 'Delivered (0)', 1: 'Returned / RTO (1)'})
                            else:
                                target_counts['Label'] = target_counts[target_col].astype(str)
                        else:
                            target_counts['Label'] = target_counts[target_col].astype(str)
                            
                        fig_target_pie = px.pie(
                            target_counts,
                            values='Count',
                            names='Label',
                            color='Label',
                            color_discrete_sequence=px.colors.qualitative.Safe,
                            hole=0.4,
                            template='plotly_white'
                        )
                        fig_target_pie.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_target_pie, use_container_width=True)
                        st.caption("Discrete Categories: Displays segment shares of each class label.")
                    else:
                        # For continuous values, plot a beautiful histogram distribution
                        fig_target_hist = px.histogram(
                            filtered_df,
                            x=target_col,
                            marginal="box",
                            color_discrete_sequence=['#4f46e5'],
                            template='plotly_white',
                            title=f"Value Distribution of `{target_col}`"
                        )
                        fig_target_hist.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig_target_hist, use_container_width=True)
                        st.caption("Continuous Numeric Distribution: Displays value ranges, counts, and a boxplot summary.")
                else:
                    st.info("Target column not present in filtered data.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with row1_2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.subheader("📝 Missing Values Profile")
                
                total_missing = filtered_df.isnull().sum().sum()
                if total_missing == 0:
                    st.markdown("""
                    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 1.2rem; color: #166534; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0 0 0.5rem 0; font-weight: 700;">🎉 Perfect Data Quality!</h4>
                        <p style="margin: 0; font-size: 0.95rem; line-height: 1.5;">All variables in your active dataset are 100% complete. No missing or null values were identified.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("Breakdown: Every column contains 0% missing values.")
                else:
                    # Calculate missing percentages
                    missing_pct = (filtered_df.isnull().sum() / len(filtered_df)) * 100
                    # Filter columns with missing values > 0 to keep chart clean
                    missing_pct = missing_pct[missing_pct > 0]
                    
                    if len(missing_pct) > 0:
                        missing_df = pd.DataFrame({
                            'Column': missing_pct.index,
                            'Missing (%)': missing_pct.values
                        }).sort_values(by='Missing (%)', ascending=True)
                        
                        fig_missing = px.bar(
                            missing_df,
                            x='Missing (%)',
                            y='Column',
                            orientation='h',
                            color='Missing (%)',
                            color_continuous_scale='reds',
                            range_x=[0, 100],
                            template='plotly_white'
                        )
                        fig_missing.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=30, b=20),
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_missing, use_container_width=True)
                    else:
                        st.info("No missing values in filtered subset.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Feature Importance Diagnostics (Full-width for readability)
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("📊 Predictive Feature Weights (Feature Influence)")
            st.write("Identify which variables have the highest decision weight in predicting the target:")
            
            if st.session_state['active_model'] is not None:
                active = st.session_state['active_model']
                agg_importances = get_clean_aggregated_importances(active)
                if len(agg_importances) > 0:
                    # Sort importances
                    imp_items = sorted(agg_importances.items(), key=lambda x: x[1], reverse=True)
                    imp_df = pd.DataFrame(imp_items, columns=['Feature Name', 'Predictive Influence'])
                    imp_df['Influence (%)'] = np.round(imp_df['Predictive Influence'] * 100, 1)
                    
                    fig_imp = px.bar(
                        imp_df,
                        x='Influence (%)',
                        y='Feature Name',
                        orientation='h',
                        color='Influence (%)',
                        color_continuous_scale='blues',
                        template='plotly_white',
                        labels={'Influence (%)': 'Influence (%)', 'Feature Name': 'Feature'}
                    )
                    fig_imp.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=30, b=20),
                        coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.markdown("""
                    💡 **How to interpret this chart:**  
                    The variables at the top of the list (with the longest bars) have the highest impact on predictions. 
                    For instance, a feature with **20% influence** has double the predictive weight of a feature with **10% influence** when the model scores transactions.
                    """)
                else:
                    st.info("No feature importance mapped in active model.")
            else:
                st.info("Please train/load a model to display feature importance.")
            st.markdown('</div>', unsafe_allow_html=True)
                
            # --- Dynamic Graphical Explorer ---
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("🎛️ Dynamic Analysis Explorer")
            st.write("Generate customized graphs by choosing variables, color groupings, and plot types:")
            
            exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
            
            with exp_col1:
                chart_type = st.selectbox(
                    "Plot Type:",
                    ["Bar Chart", "Pie Chart", "Line Chart", "Box Plot", "Histogram"]
                )
            with exp_col2:
                x_var = st.selectbox(
                    "X-Axis Variable:",
                    sorted(list(filtered_df.columns)),
                    index=0
                )
            with exp_col3:
                # Optional Y axis variable (default is Count)
                y_options = ["None (Count)"] + sorted([c for c in filtered_df.columns if c != x_var])
                y_var_sel = st.selectbox(
                    "Y-Axis Variable (Optional):",
                    y_options,
                    index=0
                )
                y_var = None if y_var_sel == "None (Count)" else y_var_sel
            with exp_col4:
                # Color grouping column selector
                color_options = ["None"] + sorted([c for c in filtered_df.columns if df_report[c].nunique() < 15])
                color_var_sel = st.selectbox(
                    "Color / Group By (Optional):",
                    color_options,
                    index=0
                )
                color_var = None if color_var_sel == "None" else color_var_sel
                
            # Plot dynamic explorer chart
            try:
                if chart_type == "Bar Chart":
                    if y_var is not None:
                        # Aggregated bar chart
                        bar_agg = filtered_df.groupby([x_var] + ([color_var] if color_var else [])).mean(numeric_only=True).reset_index()
                        fig_custom = px.bar(
                            bar_agg,
                            x=x_var,
                            y=y_var,
                            color=color_var,
                            barmode='group',
                            title=f"Bar Chart: Mean `{y_var}` by `{x_var}`",
                            template='plotly_white'
                        )
                    else:
                        # Value count bar chart
                        fig_custom = px.histogram(
                            filtered_df,
                            x=x_var,
                            color=color_var,
                            barmode='group',
                            title=f"Bar Chart: Distribution of `{x_var}`",
                            template='plotly_white'
                        )
                        
                elif chart_type == "Pie Chart":
                    # Sum count of unique classes
                    pie_df = filtered_df[x_var].value_counts().reset_index()
                    pie_df.columns = [x_var, 'Count']
                    fig_custom = px.pie(
                        pie_df,
                        values='Count',
                        names=x_var,
                        title=f"Pie Chart: Shares of `{x_var}` Categories",
                        template='plotly_white'
                    )
                    
                elif chart_type == "Line Chart":
                    if y_var is not None:
                        # Sort by X axis for chronological/numerical ordering
                        line_df = filtered_df.sort_values(by=x_var)
                        fig_custom = px.line(
                            line_df,
                            x=x_var,
                            y=y_var,
                            color=color_var,
                            title=f"Line Chart: Trend of `{y_var}` by `{x_var}`",
                            template='plotly_white'
                        )
                    else:
                        st.warning("⚠️ Line charts require a Y-axis variable to track values.")
                        fig_custom = None
                        
                elif chart_type == "Box Plot":
                    if y_var is not None:
                        fig_custom = px.box(
                            filtered_df,
                            x=x_var,
                            y=y_var,
                            color=color_var,
                            title=f"Box Plot: `{y_var}` distributions by `{x_var}`",
                            template='plotly_white'
                        )
                    else:
                        fig_custom = px.box(
                            filtered_df,
                            y=x_var,
                            color=color_var,
                            title=f"Box Plot: Distributions of `{x_var}`",
                            template='plotly_white'
                        )
                        
                elif chart_type == "Histogram":
                    fig_custom = px.histogram(
                        filtered_df,
                        x=x_var,
                        y=y_var,
                        color=color_var,
                        marginal="box", # Draw boxplot overlay
                        title=f"Histogram: Distribution of `{x_var}`",
                        template='plotly_white'
                    )
                    
                if fig_custom is not None:
                    fig_custom.update_layout(height=450)
                    st.plotly_chart(fig_custom, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Could not render selected chart combination: {e}. Check if you selected a categorical column on a numerical axis.")
            st.markdown('</div>', unsafe_allow_html=True)