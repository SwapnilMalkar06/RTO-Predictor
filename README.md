# Universal Risk & RTO Predictor

This is a Machine Learning tool designed to predict Return to Origin (RTO) / packaging return risks in real-time. It enables digital merchants to inspect, predict, and intercept package return risk.

## Project Structure
- `app.py`: Streamlit-based web dashboard / risk control center interface.
- `train_model.py`: Script to train the predictive model pipeline.
- `config.py`: Centralized configuration constants (mappings, paths, settings).
- `test_functional.py`: Automated pytest suite testing system functional parameters.
- `check_columns.py`: Utility script to check details and headers of the input dataset.
- `requirements.txt`: Dependencies needed to run the application and model.
- `data/`: Folder to store the customer transactional/orders log spreadsheet.
- `models/`: Folder containing serialized model artifact configurations.

## Quick Start Setup

### 1. Set Up Virtual Environment (Recommended)
We recommend setting up a local virtual environment:
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Required Dependencies
Install all packages defined in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Place Dataset & Train the Model
1. Obtain the transactional orders spreadsheet (e.g., `amazon_returns_dataset_cleaned.xlsx`).
2. Save it in the `data/` folder as `data/amazon_returns_dataset_cleaned.xlsx`.
3. Train the model pipeline by running:
   ```bash
   python train_model.py
   ```
This will train a Random Forest classifier and output the serialized pipeline inside the `models/` directory as `models/rto_predictor_model.pkl`.

### 4. Run the Streamlit Dashboard
Launch the control center interface local server:
```bash
streamlit run app.py
```
Open the provided URL (typically `http://localhost:8501`) in your browser to interact with the dashboard.

### 5. Run Automated Tests
Verify all system components and functional parameters using pytest:
```bash
python -m pytest test_functional.py -v
```
