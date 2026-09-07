"""
Car Price Prediction with Machine Learning
=============================================
A beginner-friendly, step-by-step regression pipeline.

Dataset columns:
Car_Name, Year, Selling_Price, Present_Price, Driven_kms,
Fuel_Type, Selling_type, Transmission, Owner

Target variable: Selling_Price (the price the car was actually sold for, in lakhs)
"""

from pathlib import Path
import argparse
import sys

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ModuleNotFoundError as exc:
    missing_module = exc.name or "a required dependency"
    print(
        f"Missing dependency: {missing_module}. "
        "Install project requirements with:\n"
        "python -m pip install -r requirements.txt\n"
        "Then rerun the app.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")

def load_data(csv_path):
    """Load and validate the car dataset used by the pipeline."""
    required_columns = {
        "Car_Name", "Year", "Selling_Price", "Present_Price", "Driven_kms",
        "Fuel_Type", "Selling_type", "Transmission", "Owner",
    }
    df = pd.read_csv(csv_path)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    df = df.dropna(subset=required_columns).copy()
    if df.empty:
        raise ValueError("Dataset has no complete rows after removing missing values.")
    return df


def main(csv_path):
    # -------------------------------------------------------------------
    # STEP 1: LOAD THE DATA
    # -------------------------------------------------------------------
    df = load_data(csv_path)
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("STEP 1: Data loaded")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(), "\n")

    # -------------------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # -------------------------------------------------------------------
    # Age and usage intensity are more useful than the raw year and total
    # mileage alone. These features do not use Selling_Price, avoiding leakage.
    current_year = int(df["Year"].max()) + 1
    df["Car_Age"] = (current_year - df["Year"]).clip(lower=0)
    df["Driven_kms_per_year"] = df["Driven_kms"] / df["Car_Age"].clip(lower=1)

    # Car_Name has ~90+ unique values (too many categories for a small
    # 301-row dataset -> would cause the model to overfit / memorize).
    # Instead we extract the BRAND (first word), which generalizes much better.
    df["Brand"] = df["Car_Name"].astype(str).str.split().str[0].str.lower()

    # Present_Price is the car's current showroom (new) price. We keep it as
    # a feature because it is available when estimating a resale price.

    print("STEP 2: Feature engineering done")
    print(df[["Car_Name", "Brand", "Year", "Car_Age", "Driven_kms_per_year"]].head(), "\n")

    # -------------------------------------------------------------------
    # STEP 3: DEFINE FEATURES (X) AND TARGET (y)
    # -------------------------------------------------------------------
    features = ["Brand", "Car_Age", "Present_Price", "Driven_kms",
                "Driven_kms_per_year", "Fuel_Type", "Selling_type",
                "Transmission", "Owner"]
    target = "Selling_Price"

    X = df[features]
    y = df[target]

    categorical_cols = ["Brand", "Fuel_Type", "Selling_type", "Transmission"]
    numeric_cols = ["Car_Age", "Present_Price", "Driven_kms",
                    "Driven_kms_per_year", "Owner"]

    # -------------------------------------------------------------------
    # STEP 4: TRAIN / TEST SPLIT
    # -------------------------------------------------------------------
    # We hold back 20% of the data purely for testing, so we can measure
    # performance on cars the model has never seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"STEP 4: Train size = {len(X_train)}, Test size = {len(X_test)}\n")

    # -------------------------------------------------------------------
    # STEP 5: PREPROCESSING PIPELINE
    # -------------------------------------------------------------------
    # Models can't read text like "Petrol" or "maruti" directly, so we
    # One-Hot-Encode categorical columns (turns each category into a 0/1 column).
    # Numeric columns pass through unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough"
    )

    # -------------------------------------------------------------------
    # STEP 6: TRAIN TWO MODELS AND COMPARE
    # -------------------------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds}
        fitted_pipelines[name] = pipe

        print(f"STEP 6: {name}")
        print(f"  MAE  (avg error, in lakhs): {mae:.3f}")
        print(f"  RMSE (penalizes big misses): {rmse:.3f}")
        print(f"  R2   (variance explained):   {r2:.3f}\n")

    # -------------------------------------------------------------------
    # STEP 7: VISUALIZATIONS
    # -------------------------------------------------------------------

    # 7a. Correlation heatmap of numeric features
    plt.figure(figsize=(6, 5))
    num_df = df[["Selling_Price", "Present_Price", "Driven_kms", "Car_Age", "Owner"]]
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(plots_dir / "1_correlation_heatmap.png")
    plt.close()

    # 7b. Actual vs Predicted for both models
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (name, res) in zip(axes, results.items()):
        ax.scatter(y_test, res["preds"], alpha=0.6, edgecolor="k")
        lims = [0, max(y_test.max(), res["preds"].max()) + 2]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual Selling Price (lakhs)")
        ax.set_ylabel("Predicted Selling Price (lakhs)")
        ax.set_title(f"{name}\nR2 = {res['R2']:.3f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "2_actual_vs_predicted.png")
    plt.close()

    # 7c. Feature importance from Random Forest
    rf_pipe = fitted_pipelines["Random Forest"]
    feature_names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(12)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="steelblue")
    plt.title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(plots_dir / "3_feature_importance.png")
    plt.close()

    # 7d. Model comparison bar chart
    comp_df = pd.DataFrame(results).T[["MAE", "RMSE", "R2"]].astype(float)
    comp_df.to_csv("model_comparison.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        sns.barplot(
            x=comp_df.index, y=comp_df[metric], hue=comp_df.index,
            ax=ax, palette="viridis", legend=False
        )
        ax.set_title(metric)
        ax.tick_params(axis="x", labelrotation=15)
    plt.tight_layout()
    plt.savefig(plots_dir / "4_model_comparison.png")
    plt.close()

    best_model_name = max(results, key=lambda k: results[k]["R2"])
    print("STEP 7: All plots saved to plots/")
    print("\nDone. Best model by R2:", best_model_name)
    return {
        "df": df,
        "features": features,
        "current_year": current_year,
        "results": results,
        "pipelines": fitted_pipelines,
        "comparison": comp_df,
        "importance": imp_df,
        "best_model": best_model_name,
    }


def render_streamlit_dashboard(model_data):
    """Render the interactive dashboard when launched with Streamlit."""
    import streamlit as st

    df = model_data["df"]
    results = model_data["results"]
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
    st.title("Car Price Predictor")
    st.caption("Estimate a used car's resale price from its specifications.")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Best model", model_data["best_model"])
    metric_cols[1].metric("Random Forest R²", f"{results['Random Forest']['R2']:.3f}")
    metric_cols[2].metric("Average error", f"{results['Random Forest']['MAE']:.3f} lakhs")

    st.subheader("Predict a car price")
    input_cols = st.columns(3)
    with input_cols[0]:
        brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
        year = st.slider("Manufacture year", int(df["Year"].min()), int(df["Year"].max()))
        present_price = st.number_input(
            "Present price (lakhs)", min_value=0.1,
            value=float(df["Present_Price"].median()), step=0.1
        )
    with input_cols[1]:
        driven_kms = st.number_input(
            "Driven kilometres", min_value=0, value=int(df["Driven_kms"].median()), step=1000
        )
        fuel_type = st.selectbox("Fuel type", sorted(df["Fuel_Type"].unique()))
        selling_type = st.selectbox("Selling type", sorted(df["Selling_type"].unique()))
    with input_cols[2]:
        transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
        owner = st.number_input("Previous owners", min_value=0, max_value=3, value=0, step=1)

    if st.button("Predict price", type="primary"):
        car_age = max(model_data["current_year"] - year, 0)
        input_row = pd.DataFrame([{
            "Brand": brand,
            "Car_Age": car_age,
            "Present_Price": present_price,
            "Driven_kms": driven_kms,
            "Driven_kms_per_year": driven_kms / max(car_age, 1),
            "Fuel_Type": fuel_type,
            "Selling_type": selling_type,
            "Transmission": transmission,
            "Owner": owner,
        }], columns=model_data["features"])
        prediction = model_data["pipelines"][model_data["best_model"]].predict(input_row)[0]
        st.success(f"Estimated selling price: {prediction:.2f} lakhs")

    st.subheader("Model evaluation")
    st.dataframe(model_data["comparison"].round(3), use_container_width=True)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.image("plots/2_actual_vs_predicted.png", caption="Actual vs predicted prices")
    with chart_cols[1]:
        st.image("plots/3_feature_importance.png", caption="Most important features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train car price regression models.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "car data.csv",
        help="Path to a car data CSV file.",
    )
    args = parser.parse_args()
    model_data = main(args.csv_path)
    if "streamlit" in sys.modules:
        render_streamlit_dashboard(model_data)