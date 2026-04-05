# FIFA World Cup 2026 Prediction Project

## Project Goal
ML model to predict the FIFA WC 2026 winner using historical match data,
FIFA rankings, squad market values, and engineered features.

## Tech Stack
- Python 3.11+
- pandas, scikit-learn, xgboost, lightgbm
- Streamlit (dashboard)
- snake_case variables, Google-style docstrings, type hints on all functions

## Project Structure
data/raw/          → original downloaded CSVs
data/processed/    → cleaned, merged datasets
notebooks/         → EDA only
src/
  ingestion/       → data downloading scripts
  features/        → feature engineering
  models/          → training, evaluation
  dashboard/       → Streamlit app

## Commands
- Run pipeline: python src/main.py
- Run dashboard: streamlit run src/dashboard/app.py
- Run tests: pytest tests/

## Rules
- Always write modular functions, never monolithic scripts
- Every data transformation must be reproducible (set random seeds)
- Save processed data to data/processed/ after each step
- Never hardcode paths — use config.yaml