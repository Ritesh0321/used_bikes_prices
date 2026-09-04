
import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

RANDOM_STATE = 42
CURRENT_YEAR = datetime.now().year

st.set_page_config(
    page_title="MotoMarket — XGBoost Used Bike Analytics",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 10% 0%, rgba(59,130,246,.14), transparent 30%),
      radial-gradient(circle at 90% 10%, rgba(168,85,247,.12), transparent 30%),
      #080b12;
    color:#f8fafc;
}
.block-container {max-width:1450px; padding-top:2rem; padding-bottom:4rem;}
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0d111b 0%,#090c13 100%);
    border-right:1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] * {color:#e5e7eb;}
.hero {
    padding:28px 32px; border-radius:24px; margin-bottom:25px;
    background:linear-gradient(135deg,rgba(30,41,59,.95),rgba(15,23,42,.92));
    border:1px solid rgba(255,255,255,.08);
    box-shadow:0 20px 60px rgba(0,0,0,.25);
}
.hero-title {font-size:42px;font-weight:800;letter-spacing:-1.5px;margin-bottom:4px;}
.hero-subtitle {color:#94a3b8;font-size:16px;}
.badge {
    display:inline-block;padding:6px 12px;margin-bottom:14px;border-radius:999px;
    background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.25);
    color:#60a5fa;font-size:12px;font-weight:700;
}
.kpi {
    padding:20px;min-height:125px;border-radius:18px;
    background:linear-gradient(145deg,rgba(30,41,59,.80),rgba(15,23,42,.70));
    border:1px solid rgba(255,255,255,.07);
    box-shadow:0 12px 35px rgba(0,0,0,.18);
}
.kpi-label {color:#94a3b8;font-size:13px;font-weight:600;}
.kpi-value {font-size:28px;font-weight:800;margin-top:8px;}
.section-title {font-size:22px;font-weight:800;margin-top:28px;margin-bottom:4px;}
.section-subtitle {color:#64748b;font-size:13px;margin-bottom:15px;}
.prediction {
    padding:30px;border-radius:22px;
    background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(124,58,237,.16));
    border:1px solid rgba(96,165,250,.20);
}
.prediction-price {font-size:42px;font-weight:800;color:#60a5fa;}
#MainMenu, footer {visibility:hidden;}
header {background:transparent!important;}
</style>
""", unsafe_allow_html=True)


# -------------------- Data preparation --------------------

OWNER_MAP = {
    "first owner": 1,
    "second owner": 2,
    "third owner": 3,
    "fourth owner or more": 4,
}

SCALE_COLUMNS = ["mileage", "power", "bike_age"]

def money(v):
    if pd.isna(v): return "₹0"
    v = float(v)
    if v >= 1e7: return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5: return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"

def clean_numeric(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    nums = pd.Series([s]).str.extract(r"([\d.]+(?:\s*-\s*[\d.]+)?)")[0]
    if nums.isna().iloc[0]: return np.nan
    s = nums.iloc[0]
    if "-" in s:
        a,b = [float(x.strip()) for x in s.split("-",1)]
        return (a+b)/2
    try: return float(s)
    except: return np.nan

def clean_kms(v):
    if pd.isna(v): return np.nan
    s = str(v).lower().replace(",","")
    m = pd.Series([s]).str.extract(r"([\d.]+)")[0].iloc[0]
    try: return float(m)
    except: return np.nan

def prepare(raw):
    df = raw.copy()
    df.columns = df.columns.astype(str).str.strip().str.lower()

    rename = {
        "model_name":"bike_model",
        "location":"city",
        "mileage (in kmpl)":"mileage",
        "power (in bhp)":"power",
    }
    df.rename(columns=rename, inplace=True)

    if "brand" not in df.columns:
        df["brand"] = df["bike_model"].astype(str).str.split().str[0]

    required = [
        "bike_model","model_year","kms_driven","owner","city",
        "mileage","power","price","brand"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["kms_driven"] = df["kms_driven"].apply(clean_kms)
    df["mileage"] = df["mileage"].apply(clean_numeric)
    df["power"] = df["power"].apply(clean_numeric)

    for c in ["bike_model","owner","city","brand"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    df = df.dropna(subset=required).copy()
    df = df[(df["price"] > 0) &
            (df["model_year"] >= 1990) &
            (df["model_year"] <= CURRENT_YEAR) &
            (df["kms_driven"] >= 0) &
            (df["mileage"] > 0) &
            (df["power"] > 0)].drop_duplicates()

    df["bike_age"] = (CURRENT_YEAR - df["model_year"]).clip(lower=0)
    df["owner_encoded"] = (
        df["owner"].map(OWNER_MAP)
        .fillna(2)
        .astype(int)
    )

    city_counts = df["city"].value_counts()
    rare = city_counts[city_counts < 10].index
    df["city_model"] = df["city"].replace(rare, "others")

    return df.reset_index(drop=True)


# -------------------- XGBoost pipeline --------------------

@st.cache_data(show_spinner=False)
def train_xgb(csv_bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    df = prepare(raw)

    # Same broad feature engineering used in the supplied project:
    # owner encoding, rare-city grouping, one-hot city, bike age,
    # log KMs, RobustScaler, log target.
    model_df = pd.get_dummies(
        df,
        columns=["city_model"],
        drop_first=True,
        dtype=float,
    )

    model_df["kms_driven_log"] = np.log1p(model_df["kms_driven"])
    model_df["price_log"] = np.log1p(model_df["price"])

    scaler = RobustScaler()
    model_df[SCALE_COLUMNS] = scaler.fit_transform(model_df[SCALE_COLUMNS])

    drop_cols = [
        "price","price_log","bike_model","owner","city",
        "brand","city_model"
    ]
    X = model_df.drop(columns=drop_cols, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    y = model_df["price_log"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE
    )

    # XGBoost configuration from the supplied project.
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=.8,
        colsample_bytree=.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    pred = np.maximum(0, np.expm1(pred_log))
    actual = np.maximum(0, np.expm1(y_test))

    metrics = {
        "r2": r2_score(actual, pred),
        "mae": mean_absolute_error(actual, pred),
        "rmse": np.sqrt(mean_squared_error(actual, pred)),
    }

    residuals = pd.DataFrame({
        "actual": actual.values,
        "predicted": pred,
    })
    residuals["error"] = residuals["predicted"] - residuals["actual"]
    residuals["abs_error"] = residuals["error"].abs()

    importance = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return df, model, scaler, list(X.columns), metrics, residuals, importance


# -------------------- UI --------------------

st.sidebar.markdown("## 🏍️ MotoMarket")
st.sidebar.caption("XGBoost used-bike price intelligence")

uploaded = st.sidebar.file_uploader(
    "Upload bike dataset", type=["csv"]
)

default_path = st.sidebar.text_input("Or dataset path", "bikes.csv")

raw_bytes = None
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
else:
    try:
        with open(default_path, "rb") as f:
            raw_bytes = f.read()
    except:
        pass

if raw_bytes is None:
    st.markdown("""
    <div class="hero">
      <div class="badge">🏍️ XGBOOST USED BIKE INTELLIGENCE</div>
      <div class="hero-title">MotoMarket Analytics</div>
      <div class="hero-subtitle">
        Explore distributions, relationships, depreciation, market segments,
        model performance, feature importance and XGBoost price predictions.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Upload the project's `bikes.csv` from the sidebar.")
    st.markdown("""
    **Expected columns:** `model_name`, `model_year`, `kms_driven`,
    `owner`, `location`, `mileage`, `power`, `price`.

    The app also accepts the cleaned names used in the dashboard reference:
    `bike_model`, `city`, `mileage`, `power`, etc.
    """)
    st.stop()

try:
    df, model, scaler, feature_cols, metrics, residuals, importance = train_xgb(raw_bytes)
except Exception as e:
    st.error(f"Could not prepare/train the model: {e}")
    st.stop()

# Filters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Market Filters")

brands = sorted(df["brand"].unique())
cities = sorted(df["city"].unique())
owners = sorted(df["owner"].unique())

sel_brands = st.sidebar.multiselect("Brand", brands, default=brands)
sel_cities = st.sidebar.multiselect("City", cities, default=cities)
sel_owners = st.sidebar.multiselect("Owner type", owners, default=owners)

yr_lo, yr_hi = int(df.model_year.min()), int(df.model_year.max())
sel_years = st.sidebar.slider("Model year", yr_lo, yr_hi, (yr_lo, yr_hi))

p_lo, p_hi = int(df.price.min()), int(df.price.max())
sel_price = st.sidebar.slider("Price range", p_lo, p_hi, (p_lo, p_hi))

filtered = df[
    df.brand.isin(sel_brands) &
    df.city.isin(sel_cities) &
    df.owner.isin(sel_owners) &
    df.model_year.between(*sel_years) &
    df.price.between(*sel_price)
].copy()

if filtered.empty:
    st.warning("No bikes match the current filters.")
    st.stop()

# Header
st.markdown("""
<div class="hero">
  <div class="badge">LIVE MARKET VIEW · XGBOOST MODEL</div>
  <div class="hero-title">🏍️ MotoMarket</div>
  <div class="hero-subtitle">
    Used Bike Price Intelligence Dashboard<br>
    Discover market structure, price relationships and what the XGBoost model learned.
  </div>
</div>
""", unsafe_allow_html=True)

# KPIs
k = st.columns(6)
kpis = [
    ("🏍️","BIKES",f"{len(filtered):,}"),
    ("💰","AVG PRICE",money(filtered.price.mean())),
    ("🎯","MEDIAN",money(filtered.price.median())),
    ("🛣️","AVG KMS",f"{filtered.kms_driven.mean():,.0f}"),
    ("📈","MODEL R²",f"{metrics['r2']:.3f}"),
    ("🎯","MAE",money(metrics["mae"])),
]
for col,(icon,label,value) in zip(k,kpis):
    with col:
        st.markdown(
            f'<div class="kpi"><div>{icon}</div><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True
        )

# ================= MARKET OVERVIEW =================
st.markdown('<div class="section-title">📊 Market Overview</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Distribution and segment-level pricing across the filtered market.</div>', unsafe_allow_html=True)

c1,c2 = st.columns(2)

with c1:
    fig = px.histogram(
        filtered, x="price", nbins=40,
        title="Price Distribution",
        labels={"price":"Price (₹)"}
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=430)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    brand = (filtered.groupby("brand",as_index=False)
             .agg(avg_price=("price","mean"), listings=("price","count"))
             .sort_values("avg_price",ascending=False).head(15))
    fig = px.bar(brand,x="avg_price",y="brand",orientation="h",
                 title="Average Price by Brand",
                 labels={"avg_price":"Average Price (₹)","brand":""},
                 text="listings")
    fig.update_traces(texttemplate="%{text} listings",textposition="outside")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",height=430)
    st.plotly_chart(fig,use_container_width=True)

# ================= RELATIONSHIPS =================
st.markdown('<div class="section-title">📈 Price Relationships</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Visualize how mileage, age, power and usage relate to resale value.</div>', unsafe_allow_html=True)

sample = filtered.sample(min(2500,len(filtered)),random_state=42)
cols = st.columns(2)

charts = [
    ("kms_driven","Price vs Kilometres Driven","Kilometres Driven","price"),
    ("model_year","Price vs Model Year","Model Year","price"),
    ("mileage","Price vs Mileage","Mileage (kmpl)","price"),
    ("power","Price vs Engine Power","Power (bhp)","price"),
]
for i,(x,title,xlab,ylab) in enumerate(charts):
    with cols[i%2]:
        fig = px.scatter(
            sample,x=x,y="price",color="owner",
            hover_data=["bike_model","brand","city","model_year"],
            title=title,labels={x:xlab,"price":"Price (₹)"},opacity=.65
        )
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",height=430)
        st.plotly_chart(fig,use_container_width=True)

# Correlation
st.markdown('<div class="section-title">🔗 Numeric Correlations</div>', unsafe_allow_html=True)
corr_cols = ["price","model_year","kms_driven","mileage","power","bike_age"]
corr = filtered[corr_cols].corr()

fig = px.imshow(
    corr, text_auto=".2f", aspect="auto",
    title="Correlation Matrix"
)
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",height=550)
st.plotly_chart(fig,use_container_width=True)

# ================= MARKET INTELLIGENCE =================
st.markdown('<div class="section-title">🏙️ Market Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Compare geography, ownership and model-level pricing.</div>', unsafe_allow_html=True)

c1,c2 = st.columns(2)
with c1:
    city_stats = (filtered.groupby("city",as_index=False)
                  .agg(avg_price=("price","mean"),count=("price","size"))
                  .sort_values("count",ascending=False).head(15))
    fig = px.bar(city_stats,x="city",y="count",title="Listings by City",
                 labels={"count":"Number of Bikes","city":"City"})
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",height=430)
    st.plotly_chart(fig,use_container_width=True)

with c2:
    owner_stats = (filtered.groupby("owner",as_index=False)
                   .agg(avg_price=("price","mean"),count=("price","size")))
    fig = px.bar(owner_stats,x="owner",y="avg_price",title="Average Price by Owner Type",
                 labels={"avg_price":"Average Price (₹)","owner":"Owner"})
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",height=430)
    st.plotly_chart(fig,use_container_width=True)

# Top models
st.markdown('<div class="section-title">🏆 Top Models</div>', unsafe_allow_html=True)
top = (filtered.groupby(["bike_model","brand"],as_index=False)
       .agg(avg_price=("price","mean"),median_price=("price","median"),
            listings=("price","size"),avg_kms=("kms_driven","mean"))
       .sort_values("avg_price",ascending=False).head(20))
top_display = top.copy()
top_display["avg_price"] = top_display.avg_price.round().astype(int)
top_display["median_price"] = top_display.median_price.round().astype(int)
top_display["avg_kms"] = top_display.avg_kms.round().astype(int)
top_display.columns = ["Bike Model","Brand","Avg Price","Median Price","Listings","Avg KMs"]
st.dataframe(top_display,use_container_width=True,hide_index=True)

# Depreciation
st.markdown('<div class="section-title">⏳ Depreciation View</div>', unsafe_allow_html=True)
age = (filtered.groupby("bike_age",as_index=False)
       .agg(avg_price=("price","mean"),median_price=("price","median"),listings=("price","size")))
fig = px.line(age,x="bike_age",y="avg_price",markers=True,
              title="Average Resale Price by Bike Age",
              labels={"bike_age":"Bike Age (years)","avg_price":"Average Price (₹)"})
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",height=430)
st.plotly_chart(fig,use_container_width=True)

# ================= MODEL PERFORMANCE =================
st.markdown('<div class="section-title">🤖 XGBoost Model Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Evaluation on the held-out 20% test set, using price after inverse log transformation.</div>', unsafe_allow_html=True)

m1,m2,m3,m4 = st.columns(4)
m1.metric("R²",f"{metrics['r2']:.4f}")
m2.metric("MAE",money(metrics["mae"]))
m3.metric("RMSE",money(metrics["rmse"]))
m4.metric("Test rows",f"{len(residuals):,}")

c1,c2 = st.columns(2)
with c1:
    fig = px.scatter(residuals,x="actual",y="predicted",
                     title="Actual vs Predicted Price",
                     labels={"actual":"Actual Price (₹)","predicted":"Predicted Price (₹)"},
                     opacity=.65)
    mn = min(residuals.actual.min(),residuals.predicted.min())
    mx = max(residuals.actual.max(),residuals.predicted.max())
    fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",
                             name="Perfect prediction"))
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",height=450)
    st.plotly_chart(fig,use_container_width=True)

with c2:
    fig = px.histogram(residuals,x="error",nbins=40,
                       title="Prediction Error Distribution",
                       labels={"error":"Predicted − Actual (₹)"})
    fig.add_vline(x=0,line_dash="dash")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",height=450)
    st.plotly_chart(fig,use_container_width=True)

# Residuals vs actual
fig = px.scatter(residuals,x="actual",y="error",
                 title="Residuals vs Actual Price",
                 labels={"actual":"Actual Price (₹)","error":"Residual (₹)"},
                 opacity=.55)
fig.add_hline(y=0,line_dash="dash")
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",height=430)
st.plotly_chart(fig,use_container_width=True)

# Feature importance
st.markdown('<div class="section-title">🌲 What the XGBoost Model Learned</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Top features by XGBoost gain/importance score.</div>', unsafe_allow_html=True)

fi = importance.head(20).sort_values("importance")
fig = px.bar(fi,x="importance",y="feature",orientation="h",
             title="Top 20 Feature Importances",
             labels={"importance":"Importance","feature":""})
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)",height=650)
st.plotly_chart(fig,use_container_width=True)

# Model configuration
with st.expander("🔧 XGBoost configuration & methodology"):
    st.write({
        "algorithm":"XGBRegressor",
        "n_estimators":200,
        "learning_rate":0.05,
        "max_depth":6,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "random_state":42,
        "test_size":0.20,
        "target":"log1p(price)",
        "inverse_transform":"expm1",
        "scaled_features":"mileage, power, bike_age",
        "engineered_features":"owner_encoded, rare-city one-hot features, bike_age, log1p(kms_driven)",
    })

# ================= PREDICTOR =================
st.markdown('<div class="section-title">🔮 Used Bike Price Predictor</div>', unsafe_allow_html=True)
st.markdown("<div class='section-subtitle'>Use the trained XGBoost model to estimate a bike's market value.</div>", unsafe_allow_html=True)

left,right = st.columns([2,1],gap="large")

with left:
    a,b = st.columns(2)
    with a:
        pred_kms = st.number_input("Kilometres driven",0.0,500000.0,19000.0,500.0)
        pred_mileage = st.number_input("Mileage (kmpl)",1.0,150.0,35.0,.5)
        pred_owner = st.selectbox("Owner type",sorted(df.owner.unique()))
    with b:
        pred_power = st.number_input("Power (bhp)",.1,300.0,20.0,.5)
        pred_year = st.number_input("Model year",int(df.model_year.min()),CURRENT_YEAR,2020,1)
        pred_city = st.selectbox("City",sorted(df.city.unique()))

    predict = st.button("✨ Estimate Market Price",use_container_width=True,type="primary")

    if predict:
        row = pd.DataFrame([{
            "model_year":pred_year,
            "kms_driven":pred_kms,
            "owner":pred_owner,
            "city":pred_city,
            "mileage":pred_mileage,
            "power":pred_power,
            "brand":df.brand.mode().iloc[0],
            "bike_model":"prediction_input",
            "price":1.0,
        }])

        # Recreate training representation.
        row["bike_age"] = CURRENT_YEAR - row["model_year"]
        row["owner_encoded"] = row["owner"].map(OWNER_MAP).fillna(2).astype(int)
        row["city_model"] = row["city"].where(row["city"].isin(df["city"].unique()),"others")

        # Construct the exact feature columns from the training data.
        # Recreate all one-hot columns and fill missing columns with zero.
        all_df = pd.concat([df.drop(columns=["price"]), row.drop(columns=["price"])],
                           ignore_index=True)
        all_df = pd.get_dummies(all_df,columns=["city_model"],drop_first=True,dtype=float)
        all_df["kms_driven_log"] = np.log1p(all_df["kms_driven"])
        all_df[SCALE_COLUMNS] = scaler.transform(all_df[SCALE_COLUMNS])
        sample = all_df.iloc[[-1]].drop(
            columns=["bike_model","owner","city","brand","city_model"],
            errors="ignore"
        )
        sample = sample.reindex(columns=feature_cols,fill_value=0)
        sample = sample.apply(pd.to_numeric,errors="coerce").fillna(0).astype(float)

        prediction = max(0,float(np.expm1(model.predict(sample)[0])))

        st.markdown(
            f'<div class="prediction"><div>Estimated market value</div>'
            f'<div class="prediction-price">{money(prediction)}</div>'
            f'<div>Powered by the trained XGBoost model</div></div>',
            unsafe_allow_html=True
        )
        r1,r2,r3 = st.columns(3)
        r1.metric("Estimate",money(prediction))
        r2.metric("Approx. lower band",money(prediction*.90))
        r3.metric("Approx. upper band",money(prediction*1.10))
        st.caption("±10% is a simple market band, not a statistical confidence interval.")

with right:
    st.markdown("### 📌 Model Summary")
    st.metric("R²",f"{metrics['r2']:.2%}")
    st.metric("MAE",money(metrics["mae"]))
    st.metric("RMSE",money(metrics["rmse"]))
    st.info("The model predicts log(price), then converts the result back to rupees.")

# Data
st.markdown('<div class="section-title">📋 Explore Listings</div>', unsafe_allow_html=True)
display_cols = ["bike_model","brand","model_year","kms_driven","owner","city","mileage","power","price"]
st.dataframe(filtered[display_cols],use_container_width=True,hide_index=True)

st.download_button(
    "⬇️ Download Filtered Dataset",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_used_bikes.csv",
    "text/csv"
)

st.markdown("""
<div style="text-align:center;color:#475569;margin-top:50px;padding:20px;border-top:1px solid rgba(255,255,255,.05)">
🏍️ <b>MotoMarket Analytics</b><br>
Used Bike Price Intelligence · XGBoost
</div>
""", unsafe_allow_html=True)
