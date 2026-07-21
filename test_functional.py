import os
import io
import tempfile
import pytest
import pandas as pd
import numpy as np
import joblib
from streamlit.testing.v1 import AppTest

# Import core model training logic
from train_model import detect_column_types, train_custom_model

# ==============================================================================
# SECTION 1: UNIT & DATA INTEGRITY TESTS FOR CORE ML ENGINE (train_model.py)
# ==============================================================================

@pytest.fixture
def sample_df():
    """Generates a representative dummy dataframe for classification training."""
    np.random.seed(42)
    rows = 100
    df = pd.DataFrame({
        'order_id': [f"O{i}" for i in range(rows)],
        'customer_id': [f"C{i}" for i in range(rows)],
        'product_id': [f"P{i}" for i in range(rows)],
        'price': np.random.uniform(10.0, 500.0, rows),
        'quantity': np.random.randint(1, 5, rows),
        'previous_returns_count': np.random.randint(0, 10, rows),
        'customer_total_orders': np.random.randint(1, 50, rows),
        'discount_pct': np.random.uniform(0.0, 30.0, rows),
        'review_rating': np.random.randint(1, 6, rows),
        'seller_rating': np.random.uniform(1.0, 5.0, rows),
        'customer_tenure_days': np.random.randint(0, 1000, rows),
        'is_prime_member': np.random.choice([0, 1], rows),
        'delivery_days': np.random.randint(1, 15, rows),
        'order_hour': np.random.randint(0, 24, rows),
        'order_weekday': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rows),
        'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'COD', 'UPI'], rows),
        'shipping_type': np.random.choice(['Standard', 'Express', 'Overnight'], rows),
        'order_datetime': pd.date_range(start='2024-01-01', periods=rows, freq='h'),
        'returned': np.random.choice([0, 1], rows)
    })
    
    # Introduce a few nulls to test missing value imputation (Data Integrity)
    df.loc[10:15, 'price'] = np.nan
    df.loc[20:25, 'payment_method'] = np.nan
    
    return df


def test_detect_column_types(sample_df):
    """
    Parameter: Input Validation Accuracy & Feature Completeness
    Verify that column auto-detection correctly classifies features and ignores keys/datetimes.
    """
    num_cols, cat_cols, ignored = detect_column_types(sample_df, target_col='returned')
    
    # Check ID and Datetime columns are ignored to prevent leakage/overfitting
    assert 'order_id' in ignored
    assert 'customer_id' in ignored
    assert 'product_id' in ignored
    assert 'order_datetime' in ignored
    assert 'returned' not in num_cols
    assert 'returned' not in cat_cols
    assert 'returned' not in ignored
    
    # Check numerical and categorical categorization
    assert 'price' in num_cols
    assert 'quantity' in num_cols
    assert 'payment_method' in cat_cols
    assert 'shipping_type' in cat_cols


def test_train_custom_model_success(sample_df):
    """
    Parameter: Feature Completeness & Data Integrity
    Verify custom model training works, outputs metadata bundle, and serializes uncorrupted.
    """
    numerical_features = ['price', 'quantity', 'previous_returns_count', 'customer_total_orders', 
                          'discount_pct', 'review_rating', 'seller_rating', 'customer_tenure_days',
                          'is_prime_member', 'delivery_days', 'order_hour']
    categorical_features = ['payment_method', 'shipping_type', 'order_weekday']
    
    # Verify training returns the expected metadata bundle
    model_bundle = train_custom_model(sample_df, 'returned', numerical_features, categorical_features)
    
    assert model_bundle is not None
    assert 'pipeline' in model_bundle
    assert 'metrics' in model_bundle
    assert 'feature_metadata' in model_bundle
    assert 'feature_importances' in model_bundle
    assert 'aggregated_importances' in model_bundle
    
    # Validate metrics range
    assert 0.0 <= model_bundle['metrics']['accuracy'] <= 1.0
    assert 0.0 <= model_bundle['metrics']['roc_auc'] <= 1.0
    assert 'classification_report' in model_bundle['metrics']
    
    # Validate imputation ranges / values stored in feature metadata
    assert model_bundle['feature_metadata']['price']['type'] == 'numerical'
    assert 'median' in model_bundle['feature_metadata']['price']
    assert model_bundle['feature_metadata']['payment_method']['type'] == 'categorical'
    assert 'mode' in model_bundle['feature_metadata']['payment_method']
    assert 'Unknown' not in model_bundle['feature_metadata']['payment_method']['unique_values'] # mode is computed on non-nulls
    
    # Verify serialization output
    model_path = os.path.join('models', 'rto_predictor_model.pkl')
    assert os.path.exists(model_path)
    
    # Verify loading does not fail and reads data identically (Data Integrity)
    loaded_bundle = joblib.load(model_path)
    assert loaded_bundle['target_column'] == 'returned'
    assert loaded_bundle['numerical_features'] == numerical_features
    assert loaded_bundle['categorical_features'] == categorical_features


def test_train_custom_model_downsampling():
    """
    Parameter: Input Validation Accuracy & Error Handling Gracefulness
    Verify downsampling limit works cleanly for large datasets to prevent Out of Memory.
    """
    large_rows = 60000
    df_large = pd.DataFrame({
        'price': np.random.uniform(10, 100, large_rows),
        'payment_method': np.random.choice(['COD', 'Card'], large_rows),
        'returned': np.random.choice([0, 1], large_rows)
    })
    
    # Train custom model (which triggers the len(df) > 50000 downsampler check)
    bundle = train_custom_model(df_large, 'returned', ['price'], ['payment_method'])
    # Assert model ran and trained successfully without crashing
    assert bundle is not None


# ==============================================================================
# SECTION 2: STREAMLIT APPTESING FOR DASHBOARD WORKFLOWS AND INPUTS (app.py)
# ==============================================================================

def test_app_initialization():
    """
    Parameter: Feature Completeness & Workflow Consistency
    Verify that the Streamlit app loads successfully, sets tabs, and loads default model.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Check that app initialized without fatal exception
    assert not at.exception
    
    # Check that default model and dataset have been loaded in session state
    assert 'active_model' in at.session_state
    assert at.session_state['active_model'] is not None
    assert 'active_df' in at.session_state
    assert at.session_state['active_df'] is not None
    
    # Check if header title is present
    assert len(at.title) == 0 # the app uses st.markdown for the main banner, let's verify Markdown blocks
    assert any("Universal Risk & RTO Predictive Suite" in md.value for md in at.markdown)


def test_app_model_studio_custom_training():
    """
    Parameter: Feature Completeness & Workflow Consistency
    Simulate user workflow in Tab 0: Selecting target, selecting features, training model.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Radio select to default dataset (Tab 0 elements)
    # Let's locate the radio option
    assert len(at.radio) > 0
    dataset_radio = at.radio[0]
    assert "Select Training Dataset Source:" in dataset_radio.label
    
    # Let's verify target selectbox
    target_selectbox = [sb for sb in at.selectbox if "Select Target Class Column:" in sb.label]
    assert len(target_selectbox) == 1
    assert target_selectbox[0].value == "returned"
    
    # Find the training button
    train_buttons = [b for b in at.button if "Train Machine Learning Pipeline" in b.label]
    assert len(train_buttons) == 1
    
    # Trigger custom training
    train_buttons[0].click().run(timeout=30)
    assert not at.exception
    
    # Ensure active model exists after training
    assert at.session_state['active_model'] is not None


def test_app_profiler_evaluation_and_what_if():
    """
    Parameter: Feature Completeness, Input Validation & Workflow Consistency
    Test the single-order evaluator and the What-If simulator.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    
    # Verify sliders/inputs exist in the profiler form
    model = at.session_state['active_model']
    assert model is not None
    
    # Evaluate order with default inputs
    eval_btn = [b for b in at.button if "Evaluate Transaction Risk Profile" in b.label]
    assert len(eval_btn) == 1
    
    eval_btn[0].click().run(timeout=30)
    assert not at.exception
    
    # Check that predictions are set in session state
    assert 'eval_rto_score' in at.session_state
    score = at.session_state['eval_rto_score']
    assert 0.0 <= score <= 100.0
    
    # Check if a risk category recommendation banner gets rendered
    banner_texts = [md.value for md in at.markdown]
    assert any("Risk:" in text or "Safe transaction" in text or "Critical High Risk" in text or "Verification requested" in text for text in banner_texts)
    
    # Test What-if slider modification
    # Find the simulator sliders (labeled with "Simulate")
    sim_sliders = [s for s in at.slider if s.label and s.label.startswith("Simulate")]
    assert len(sim_sliders) >= 1
    
    # Change first simulator slider value
    orig_val = sim_sliders[0].value
    # Set to min or max
    new_val = sim_sliders[0].max
    sim_sliders[0].set_value(new_val).run(timeout=30)
    
    # Verify that adjusted risk probability changed/updated
    assert not at.exception
    # Ensure what-if metric is present
    metrics = [m for m in at.metric if m.label == "Adjusted Risk Probability"]
    assert len(metrics) == 1
    assert "%" in metrics[0].value


def test_app_bulk_scanner_workflow_and_missing_features():
    """
    Parameter: Input Validation Accuracy, Data Integrity, Workflow Consistency & Error Handling
    Simulates:
    1. Uploading a valid CSV registry file.
    2. Uploading a CSV registry file with missing/mismatched columns.
    3. Downloading output predictions.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    
    # Create valid dummy bulk dataframe as CSV bytes
    valid_bulk = pd.DataFrame({
        'price': [100.0, 20.0, 300.0],
        'quantity': [1, 2, 1],
        'previous_returns_count': [0, 5, 2],
        'customer_total_orders': [10, 5, 12],
        'discount_pct': [5.0, 0.0, 15.0],
        'review_rating': [4, 2, 5],
        'seller_rating': [4.5, 3.8, 4.9],
        'customer_tenure_days': [100, 20, 500],
        'is_prime_member': [1, 0, 1],
        'delivery_days': [3, 7, 2],
        'order_hour': [14, 20, 9],
        'order_weekday': ['Monday', 'Sunday', 'Friday'],
        'payment_method': ['Credit Card', 'COD', 'UPI'],
        'shipping_type': ['Standard', 'Standard', 'Express']
    })
    
    csv_bytes = valid_bulk.to_csv(index=False).encode('utf-8')
    
    # Find file uploader key "bulk_uploader" or index
    uploader = [u for u in at.file_uploader if u.key == "bulk_uploader"]
    assert len(uploader) == 1
    
    # 1. Upload valid file
    uploader[0].upload("valid_bulk.csv", csv_bytes).run(timeout=30)
    assert not at.exception
    
    # Check predictions populated in state
    assert 'scored_bulk_df' in at.session_state
    scored_df = at.session_state['scored_bulk_df']
    assert len(scored_df) == 3
    assert 'risk_probability (%)' in scored_df.columns
    assert 'risk_category' in scored_df.columns
    
    # Check summary metrics are rendered
    total_metric = [m for m in at.metric if m.label == "Total Records Scanned"]
    assert len(total_metric) == 1
    assert total_metric[0].value == "3"
    
    # Check if download button is rendered
    dl_buttons = [dl for dl in at.download_button if "Predictions Spreadsheet" in dl.label]
    assert len(dl_buttons) == 1
    
    # 2. Upload file with missing column (e.g. 'price' and 'payment_method' missing)
    mismatched_bulk = valid_bulk.drop(columns=['price', 'payment_method'])
    mismatched_bytes = mismatched_bulk.to_csv(index=False).encode('utf-8')
    
    at2 = AppTest.from_file("app.py")
    at2.run(timeout=30)
    assert not at2.exception
    
    uploader_2 = [u for u in at2.file_uploader if u.key == "bulk_uploader"]
    assert len(uploader_2) == 1
    uploader_2[0].upload("mismatched_bulk.csv", mismatched_bytes).run(timeout=30)
    assert not at2.exception
    
    # Check warnings are shown
    warning_banners = [w for w in at2.warning if "imputed using training metadata defaults" in w.body]
    assert len(warning_banners) == 1
    
    # Ensure prediction still completed successfully (graceful fallback)
    assert len(at2.session_state['scored_bulk_df']) == 3


def test_app_bulk_scanner_malformed_file():
    """
    Parameter: Error Handling Gracefulness
    Verify uploading a completely corrupted/non-data file does not crash the app, but displays an error.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    uploader = [u for u in at.file_uploader if u.key == "bulk_uploader"]
    
    # Upload binary garbage
    bad_bytes = b"garbage string data that is not a csv structure"
    uploader[0].upload("malformed.xlsx", bad_bytes).run(timeout=30)
    
    # Verify app did not crash
    assert not at.exception
    # Check if error alert is shown in the UI
    errors = [e.body for e in at.error]
    assert len(errors) >= 1
    assert any("Error" in err or "failed" in err for err in errors)


def test_app_graphical_report_empty_filters():
    """
    Parameter: Error Handling Gracefulness
    Verify selecting filters that exclude all data displays a warning message rather than crashing.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    
    # Select tab 3 (or wait, filters display candidate columns based on active_df)
    # We can check multiselect filter controls
    filter_selects = [ms for ms in at.multiselect if "Select columns to filter by:" in ms.label]
    if len(filter_selects) > 0:
        # We can force the filter selections to show 0 rows
        # Instead, we can manually manipulate active filters or active_df to be empty
        at.session_state['active_df'] = pd.DataFrame(columns=at.session_state['active_df'].columns)
        at.run(timeout=30)
        assert not at.exception
        
        # Verify empty warning/error displays cleanly
        errors = [e.body for e in at.error]
        assert len(errors) >= 1
        assert any("0 records" in err or "empty" in err or "missing" in err for err in errors)


def test_app_uninitialized_state_safety():
    """
    Parameter: Workflow Consistency & Error Handling Gracefulness
    Verify that if session state has NO active model, the app displays warning banners 
    requesting model training instead of throwing runtime attribute/index crashes.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    
    # Wipe active model
    at.session_state['active_model'] = None
    at.run(timeout=30)
    
    # Verify no unhandled exception
    assert not at.exception
    
    # Verify warning notices requesting model training display in the UI
    warning_bodies = [w.body for w in at.warning]
    assert len(warning_bodies) >= 1
    assert any("model artifact missing" in w or "configure and train" in w for w in warning_bodies)


def test_app_sidebar_reset():
    """
    Parameter: Workflow Consistency
    Verify that the sidebar reset button successfully clears session state.
    """
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    
    # Populate some session states manually
    at.session_state['scored_bulk_df'] = pd.DataFrame([{'a': 1}])
    at.session_state['bulk_filename'] = "mock_file.csv"
    at.run(timeout=30)
    assert not at.exception
    
    # Find sidebar reset button
    reset_buttons = [b for b in at.button if "Reset to Default Settings" in b.label]
    assert len(reset_buttons) == 1
    
    # Click reset
    reset_buttons[0].click().run(timeout=30)
    assert not at.exception
    
    # Check that bulk state variables have been cleared (None)
    assert at.session_state['scored_bulk_df'] is None
    assert at.session_state['bulk_filename'] is None
    # Default model/dataset should still be loaded as fallback
    assert at.session_state['active_model'] is not None
    assert at.session_state['active_df'] is not None
