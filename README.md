#WORKING LINK
https://retail-store-sales-prediction.streamlit.app/
# Retail Store Sales Prediction & Intelligence

A Machine Learning-based web application that predicts retail product demand and provides interactive sales analytics using Python and Streamlit.

## Project Overview

The Retail Store Sales Intelligence project helps analyze historical retail sales data and forecast the expected number of units sold for selected products.

It provides an interactive dashboard where users can configure forecasting parameters, view monthly predictions, analyze trends, and download forecast results.

## Objectives

- Predict retail product demand using Machine Learning.
- Analyze historical sales patterns.
- Generate monthly sales forecasts.
- Estimate potential revenue.
- Present insights through interactive visualizations.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Key Features

- Single-month sales prediction.
- Custom date-range forecasting.
- Monthly demand visualization.
- Estimated revenue analysis.
- Historical sales analytics.
- Top-performing product analysis.
- Dataset preview.
- CSV report download.
- Interactive dashboard with Light and Dark themes.

## Dataset Information

The project uses a retail sales dataset downloaded from Kaggle.

Dataset characteristics:
- 1,461 records
- 11 columns
- 111 products
- Historical data from 2021 to 2024

The dataset file (`sales.xlsx`) is not included in this repository. Download the original dataset from its Kaggle source, ensure its usage licence permits your intended use, and place it in the project folder with the filename `sales.xlsx`.

## Machine Learning Model

The project uses a trained Machine Learning regression model to estimate product demand.

Input features include:
- Month
- Year
- Customer Age
- Product Name
- Final Price
- Sales Event
- Inventory Pressure

The model predicts the expected number of units sold.

Estimated Revenue = Predicted Units × Product Price

**Note:** Estimated revenue is not actual revenue or profit.

## Project Structure

```text
RETAIL-STORE-SALES-PREDICTION/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── sales_prediction.py
├── sales_model.pkl
├── requirements.txt
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ARAHIMGTH/RETAIL-STORE-SALES-PREDICTION.git
```

### 2. Open the project folder

```bash
cd RETAIL-STORE-SALES-PREDICTION
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the downloaded dataset in the project folder and name it:

```text
sales.xlsx
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

## Limitations

- Predictions depend on the quality and coverage of historical data.
- Training data covers 2021–2024.
- Forecasts beyond the training period should be treated as estimates.
- Forecast assumptions remain constant across the selected date range.
- Estimated revenue does not account for operating costs or profit margins.

## Future Scope

- Integration with real-time sales data.
- Improved forecasting models.
- Automatic festival and seasonal detection.
- Advanced demand analysis.
- Cloud deployment.

## Author

**Abdur Rahim**

B.Tech – Computer Science and Engineering (Data Science & AI)

GitHub: https://github.com/ARAHIMGTH

---

*Developed as an academic Machine Learning and Data Analytics project.*
