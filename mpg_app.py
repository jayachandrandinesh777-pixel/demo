"""
MPG Predictor - Streamlit app (trains 4 models, deploys the best).

Run locally:   pip install -r requirements.txt && streamlit run mpg_app.py
Streamlit Cloud: put mpg_app.py, requirements.txt and auto-mpg.csv in the repo root.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="centered")

BASE = Path(__file__).resolve().parent
SEED = 42
NUMERIC = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
CATEGORICAL = ["origin"]  # 1 = USA, 2 = Europe, 3 = Japan
ORIGINS = {1: "USA", 2: "Europe", 3: "Japan"}


# ------------------------------------------------------------------ data
def find_csv():
    for pattern in ("auto-mpg.csv", "*auto-mpg*.csv", "*mpg*.csv"):
        hits = sorted(BASE.glob(pattern)) or sorted(Path.cwd().glob(pattern))
        if hits:
            return hits[0]
    return None


def clean(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")  # '?' -> NaN
    return df


# ------------------------------------------------------------------ ML pipeline
def build_pipeline(model):
    """Impute -> scale numerics; impute -> one-hot categoricals; then the model."""
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


@st.cache_resource(show_spinner="Training 4 models (first load only, ~1 minute)...")
def train(df):
    X, y = df[NUMERIC + CATEGORICAL], df["mpg"]  # 'car name' dropped (free text)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    candidates = {
        "Ridge Regression": (Ridge(), {"model__alpha": [0.1, 1, 10, 100]}),
        "Random Forest": (RandomForestRegressor(random_state=SEED),
                          {"model__n_estimators": [200], "model__max_depth": [None, 8, 12],
                           "model__min_samples_leaf": [1, 2]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED),
                              {"model__n_estimators": [200, 400], "model__learning_rate": [0.03, 0.1],
                               "model__max_depth": [2, 3]}),
        "SVR (RBF)": (SVR(), {"model__C": [1, 10, 50], "model__gamma": ["scale", 0.05],
                              "model__epsilon": [0.1, 0.5]}),
    }
    cv = KFold(5, shuffle=True, random_state=SEED)
    results = []
    for name, (model, grid) in candidates.items():
        gs = GridSearchCV(build_pipeline(model), grid, cv=cv,
                          scoring="neg_root_mean_squared_error", n_jobs=1).fit(X_tr, y_tr)
        pred = gs.predict(X_te)
        results.append({"Model": name,
                        "CV RMSE": round(-gs.best_score_, 3),
                        "Test RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
                        "Test MAE": round(mean_absolute_error(y_te, pred), 3),
                        "Test R²": round(r2_score(y_te, pred), 3),
                        "params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()}})
    best = min(results, key=lambda r: r["CV RMSE"])  # choose on CV, not on the test set
    final = build_pipeline(candidates[best["Model"]][0].set_params(**best["params"])).fit(X, y)
    table = pd.DataFrame(results).drop(columns="params")
    return final, best["Model"], table


# ------------------------------------------------------------------ UI
st.title("🚗 MPG Predictor")
st.caption("Predict a car's fuel efficiency (miles per gallon) from its specifications.")

csv_path = find_csv()
if csv_path is not None:
    data = clean(pd.read_csv(csv_path))
else:
    st.warning("auto-mpg.csv was not found next to the app. Upload it to continue "
               "(or add it to your GitHub repo).")
    up = st.file_uploader("Upload auto-mpg.csv", type="csv")
    if up is None:
        st.stop()
    data = clean(pd.read_csv(up))

pipeline, best_name, results = train(data)

tab_predict, tab_models, tab_data = st.tabs(["Predict", "Model comparison", "Dataset"])

with tab_predict:
    st.success(f"Model in use: **{best_name}** (best of {len(results)} algorithms)")
    c1, c2 = st.columns(2)
    cylinders = c1.number_input("Cylinders", min_value=3, max_value=8, value=4, step=1)
    displacement = c2.number_input("Displacement (cu in)", min_value=60, max_value=460, value=150, step=1)
    unknown_hp = st.checkbox("Horsepower unknown (fill automatically)")
    c3, c4 = st.columns(2)
    horsepower = c3.number_input("Horsepower", min_value=40, max_value=250, value=95, step=1,
                                 disabled=unknown_hp)
    weight = c4.number_input("Weight (lbs)", min_value=1500, max_value=5200, value=2800, step=10)
    c5, c6 = st.columns(2)
    acceleration = c5.number_input("Acceleration (0-60 mph, sec)", min_value=8.0, max_value=25.0,
                                   value=15.0, step=0.1)
    year = c6.number_input("Model year (70-82)", min_value=70, max_value=82, value=76, step=1)
    origin = st.selectbox("Origin", options=list(ORIGINS), format_func=lambda k: ORIGINS[k])

    if st.button("Predict MPG", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "cylinders": cylinders, "displacement": displacement,
            "horsepower": np.nan if unknown_hp else horsepower,  # imputed by the pipeline
            "weight": weight, "acceleration": acceleration, "model year": year, "origin": origin}])
        mpg = float(pipeline.predict(row)[0])
        m1, m2 = st.columns(2)
        m1.metric("Predicted fuel efficiency", f"{mpg:.1f} MPG")
        m2.metric("Equivalent", f"{235.215 / mpg:.1f} L/100 km" if mpg > 0 else "n/a")
    st.caption("Trained on 1970-82 cars, so results for modern vehicles are unreliable.")

with tab_models:
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index("Model")["Test RMSE"])
    st.caption("Lower RMSE/MAE is better; higher R² is better. The best model is chosen "
               "by 5-fold cross-validated RMSE on the training split, then refit on all data.")

with tab_data:
    st.write(f"{len(data)} rows; {int(data['horsepower'].isna().sum())} missing horsepower values "
             "(handled by median imputation).")
    st.dataframe(data.head(50), use_container_width=True)
