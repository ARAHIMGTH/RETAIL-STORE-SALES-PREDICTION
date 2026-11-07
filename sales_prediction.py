
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_excel("sales.xlsx")

print("\n----- FIRST 5 ROWS -----")
print(df.head())

print("\n----- DATASET INFORMATION -----")
df.info()

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())


# ==========================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ==========================================

# Monthly revenue

if {"Month", "revenue"}.issubset(df.columns):

    monthly_sales = (
        df.groupby("Month")["revenue"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(10, 5))

    monthly_sales.plot(kind="bar")

    plt.title("Monthly Sales Revenue")
    plt.xlabel("Month")
    plt.ylabel("Total Revenue")
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# Top 10 products by revenue

if {"itemDescription", "revenue"}.issubset(df.columns):

    top_products = (
        df.groupby("itemDescription")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12, 6))

    top_products.plot(kind="bar")

    plt.title("Top 10 Products by Revenue")
    plt.xlabel("Product")
    plt.ylabel("Total Revenue")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# ==========================================
# STEP 3: PREPARE MACHINE LEARNING DATA
# ==========================================

features = [
    "Month",
    "Year",
    "customer_age",
    "itemDescription",
    "final_price",
    "sales_event",
    "inventory_pressure"
]

target = "units_sold"

# Check required columns

required_columns = features + [target]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

# Remove missing values from model columns

model_df = (
    df[required_columns]
    .dropna()
    .copy()
)

# Input and target

X = model_df[features]
y = model_df[target]

# Categorical features

categorical_features = [
    "itemDescription",
    "sales_event",
    "inventory_pressure"
]

# Numerical features

numerical_features = [
    "Month",
    "Year",
    "customer_age",
    "final_price"
]

# Encode categorical data

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# STEP 5: MODEL COMPARISON
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, regressor in models.items():

    # Create pipeline

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ]
    )

    # Train model

    model.fit(X_train, y_train)

    # Predict test data

    predictions = model.predict(X_test)

    # Evaluation metrics

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model


# ==========================================
# STEP 6: DISPLAY MODEL RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print("\n----- MODEL COMPARISON RESULTS -----")

print(
    results_df.round(4).to_string(index=False)
)


# ==========================================
# STEP 7: SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\n----- BEST MODEL -----")
print(best_model_name)


# ==========================================
# STEP 8: BEST MODEL EVALUATION
# ==========================================

best_predictions = best_model.predict(X_test)

best_mae = mean_absolute_error(
    y_test,
    best_predictions
)

best_rmse = mean_squared_error(
    y_test,
    best_predictions
) ** 0.5

best_r2 = r2_score(
    y_test,
    best_predictions
)

print("\n----- BEST MODEL RESULTS -----")

print("Model:", best_model_name)
print("MAE:", round(best_mae, 2))
print("RMSE:", round(best_rmse, 2))
print("R2 Score:", round(best_r2, 4))


# ==========================================
# STEP 9: ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.7
)

# Perfect prediction reference line

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    color="red",
    label="Perfect Prediction"
)

plt.title(
    f"Actual vs Predicted - {best_model_name}"
)

plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")

plt.legend()
plt.tight_layout()
plt.show()


# ==========================================
# STEP 10: FEATURE IMPORTANCE
# ==========================================

regressor = best_model.named_steps["regressor"]

# Get encoded feature names

feature_names = (
    best_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get feature importance values

if hasattr(regressor, "feature_importances_"):

    importance = regressor.feature_importances_

elif hasattr(regressor, "coef_"):

    importance = abs(regressor.coef_)

else:

    importance = None


if importance is not None:

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(10)
    )

    print("\n----- TOP 10 IMPORTANT FEATURES -----")

    print(
        feature_importance.round(4)
        .to_string(index=False)
    )

    # Plot feature importance

    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.title("Top 10 Important Features")
    plt.xlabel("Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()


# ==========================================
# STEP 11: SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "sales_model.pkl"
)

print("\nBest model saved successfully!")
print("Saved file: sales_model.pkl")
