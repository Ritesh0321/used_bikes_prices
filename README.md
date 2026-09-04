# MotoMarket XGBoost Analytics Dashboard

This is the expanded Streamlit version requested from the reference dashboard.

It keeps the dashboard style of the uploaded app (dark MotoMarket UI, KPI cards,
Plotly charts, filters, market intelligence, predictor) but switches the ML section
to the XGBoost workflow from the supplied Used Bike Prices project.

## Includes

- Market overview and KPIs
- Price distribution
- Average price by brand
- Price vs kilometres
- Price vs model year
- Price vs mileage
- Price vs power
- Correlation matrix
- City and owner analysis
- Top bike models
- Depreciation / price by bike age
- XGBoost R² / MAE / RMSE
- Actual vs predicted chart
- Error distribution
- Residual plot
- XGBoost feature importance
- Model methodology/configuration
- Interactive XGBoost price predictor
- Filtered data table and CSV download

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put `bikes.csv` next to `app.py`, or upload it in the sidebar.

Expected original columns:
model_name, model_year, kms_driven, owner, location, mileage, power, price
