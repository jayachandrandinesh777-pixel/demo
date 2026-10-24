"""
MPG Predictor - Streamlit app (trains 4 models, deploys the best).

Run locally:   pip install -r requirements.txt && streamlit run mpg_app.py
Streamlit Cloud: put mpg_app.py, requirements.txt and auto-mpg.csv in the repo root.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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


# ------------------------------------------------------------------ car animation
CAR_HTML = """
<style>
  html,body{margin:0;background:transparent;font-family:system-ui,sans-serif}
  .scene{position:relative;height:200px;border-radius:14px;overflow:hidden;
         background:linear-gradient(#7dd3fc,#e0f2fe 70%)}
  .hills{position:absolute;left:0;bottom:48px;width:200%;height:70px;opacity:.55;
         background:radial-gradient(60px 40px at 60px 70px,#4ade80 98%,transparent) 0 0/180px 70px repeat-x;
         animation:scroll calc(var(--t)*5) linear infinite}
  .cloud{position:absolute;top:22px;left:0;width:200%;height:40px;opacity:.9;
         background:radial-gradient(28px 14px at 40px 20px,#fff 95%,transparent) 0 0/260px 40px repeat-x;
         animation:scroll calc(var(--t)*9) linear infinite}
  .road{position:absolute;left:0;right:0;bottom:0;height:52px;background:#374151}
  .lane{position:absolute;left:0;bottom:22px;width:200%;height:5px;
        background:repeating-linear-gradient(90deg,#fde68a 0 36px,transparent 36px 72px);
        animation:scroll var(--t) linear infinite}
  .car{position:absolute;left:50%;bottom:34px;width:210px;margin-left:-105px;
       animation:bob .5s ease-in-out infinite alternate}
  .wheel{transform-box:fill-box;transform-origin:center;animation:spin calc(var(--t)*.6) linear infinite}
  .smoke circle{fill:#9ca3af;opacity:0;animation:puff 1.2s ease-out infinite}
  .smoke circle:nth-child(2){animation-delay:.4s}.smoke circle:nth-child(3){animation-delay:.8s}
  .badge{position:absolute;top:10px;right:12px;background:#0008;color:#fff;padding:6px 12px;
         border-radius:20px;font-size:13px;font-weight:600}
  .bar{position:absolute;left:12px;top:12px;width:120px;height:10px;background:#0003;border-radius:6px}
  .bar i{display:block;height:100%;width:var(--fill);background:var(--c);border-radius:6px;transition:width 1s}
  @keyframes scroll{to{transform:translateX(-50%)}}
  @keyframes spin{to{transform:rotate(360deg)}}
  @keyframes bob{to{transform:translateY(-3px)}}
  @keyframes puff{0%{opacity:.7;transform:translate(0,0) scale(.5)}100%{opacity:0;transform:translate(-40px,-14px) scale(1.6)}}
</style>
<div class="scene" style="--t:__SPEED__s;--c:__COLOR__;--fill:__FILL__%">
  <div class="cloud"></div><div class="hills"></div>
  <div class="bar"><i></i></div><div class="badge">__LABEL__</div>
  <div class="road"><div class="lane"></div></div>
  <svg class="car" viewBox="0 0 210 80" xmlns="http://www.w3.org/2000/svg">
    <g class="smoke"><circle cx="10" cy="56" r="6"/><circle cx="10" cy="56" r="6"/><circle cx="10" cy="56" r="6"/></g>
    <path d="M14 56 L14 44 Q14 38 22 36 L52 32 L74 12 Q78 8 86 8 L132 8 Q140 8 146 14 L164 32 L190 38 Q198 40 198 48 L198 56 Z" fill="__COLOR__"/>
    <path d="M80 15 L118 15 L118 32 L58 32 Z" fill="#bae6fd"/>
    <path d="M124 15 L136 15 Q140 15 143 19 L156 32 L124 32 Z" fill="#bae6fd"/>
    <rect x="186" y="42" width="12" height="7" rx="3" fill="#fde047"/>
    <rect x="14" y="44" width="8" height="6" rx="2" fill="#ef4444"/>
    <g><circle cx="55" cy="58" r="17" fill="#111827"/>
      <g class="wheel"><circle cx="55" cy="58" r="9" fill="#d1d5db"/>
        <path d="M55 49V67M46 58H64" stroke="#6b7280" stroke-width="3"/></g></g>
    <g><circle cx="155" cy="58" r="17" fill="#111827"/>
      <g class="wheel"><circle cx="155" cy="58" r="9" fill="#d1d5db"/>
        <path d="M155 49V67M146 58H164" stroke="#6b7280" stroke-width="3"/></g></g>
  </svg>
</div>
"""


def car_html(mpg=None):
    """Animated car. After a prediction its colour/speed reflect fuel efficiency."""
    if mpg is None:
        color, speed, fill, label = "#3b82f6", 1.4, 0, "Enter specs and predict ➜"
    elif mpg >= 30:
        color, speed, fill, label = "#10b981", 1.0, min(mpg / 50, 1) * 100, f"{mpg:.1f} MPG · Efficient 🌿"
    elif mpg >= 20:
        color, speed, fill, label = "#f59e0b", 1.5, min(mpg / 50, 1) * 100, f"{mpg:.1f} MPG · Average"
    else:
        color, speed, fill, label = "#ef4444", 2.2, min(mpg / 50, 1) * 100, f"{mpg:.1f} MPG · Thirsty ⛽"
    return (CAR_HTML.replace("__COLOR__", color).replace("__SPEED__", str(speed))
            .replace("__FILL__", f"{fill:.0f}").replace("__LABEL__", label))


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
    anim_slot = st.empty()  # animated car, filled after the button logic below
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
        st.session_state["last_mpg"] = mpg
        m1, m2 = st.columns(2)
        m1.metric("Predicted fuel efficiency", f"{mpg:.1f} MPG")
        m2.metric("Equivalent", f"{235.215 / mpg:.1f} L/100 km" if mpg > 0 else "n/a")
    with anim_slot.container():
        components.html(car_html(st.session_state.get("last_mpg")), height=215)
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
