from datetime import date, timedelta
import json
from urllib.request import urlopen
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score

st.set_page_config(page_title="Crime Hotspot Prediction - Student Project", page_icon="🛡️", layout="wide")

INDIA_DIRECTORY_URL = "https://raw.githubusercontent.com/jayeshgupta91/Indian-States-Districts/main/states-districts.json"

@st.cache_data(ttl=86400, show_spinner=False)
def india_state_district_directory():
    """Load all-India State/UT and district options; retain a useful offline fallback."""
    fallback_states = [
        "Andaman and Nicobar Islands (UT)", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh (UT)", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu (UT)", "Delhi (NCT)", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir (UT)", "Jharkhand", "Karnataka", "Kerala", "Ladakh (UT)", "Lakshadweep (UT)", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry (UT)", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    ]
    try:
        with urlopen(INDIA_DIRECTORY_URL, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        directory = {row["state"].strip(): sorted({str(d).strip() for d in row["districts"]}) for row in payload["states"]}
        return directory, True
    except Exception:
        return {state: [] for state in fallback_states}, False

@st.cache_data
def make_demo_data():
    """Synthetic, anonymous study dataset covering all Indian States and Union Territories."""
    rng = np.random.default_rng(2026)
    all_states_areas = {
        "Andaman and Nicobar Islands (UT)": [("Port Blair", "South Andaman", 11.6233, 92.7265, 0.45, 35)],
        "Andhra Pradesh": [("Vijayawada Central", "NTR", 16.5062, 80.6480, 0.66, 39), ("Visakhapatnam Port", "Visakhapatnam", 17.6868, 83.2185, 0.84, 47)],
        "Arunachal Pradesh": [("Itanagar Capital", "Papum Pare", 27.0844, 93.6053, 0.35, 30)],
        "Assam": [("Guwahati Central", "Kamrup Metropolitan", 26.1445, 91.7362, 0.85, 52)],
        "Bihar": [("Patna Town", "Patna", 25.5941, 85.1376, 1.25, 72)],
        "Chandigarh (UT)": [("Chandigarh Sector 17", "Chandigarh", 30.7333, 76.7794, 0.60, 40)],
        "Chhattisgarh": [("Raipur City", "Raipur", 21.2514, 81.6296, 0.78, 55)],
        "Dadra and Nagar Haveli and Daman and Diu (UT)": [("Daman Fort Area", "Daman", 20.3974, 72.8328, 0.40, 32)],
        "Delhi (NCT)": [("Connaught Place", "New Delhi", 28.6304, 77.2177, 1.65, 68), ("Rohini West", "North West Delhi", 28.7041, 77.1025, 1.40, 64)],
        "Goa": [("Panaji Coastal", "North Goa", 15.4909, 73.8278, 0.55, 36)],
        "Gujarat": [("Ahmedabad Central", "Ahmedabad", 23.0225, 72.5714, 0.95, 48)],
        "Haryana": [("Gurugram Cyberhub", "Gurugram", 28.4595, 77.0266, 1.10, 50)],
        "Himachal Pradesh": [("Shimla Mall Road", "Shimla", 31.1048, 77.1734, 0.42, 28)],
        "Jammu and Kashmir (UT)": [("Srinagar City", "Srinagar", 34.0837, 74.7973, 0.65, 45)],
        "Jharkhand": [("Ranchi Town", "Ranchi", 23.3441, 85.3096, 0.88, 58)],
        "Karnataka": [("Bengaluru Indiranagar", "Bengaluru Urban", 12.9716, 77.5946, 1.35, 54)],
        "Kerala": [("Kochi Marine Drive", "Ernakulam", 9.9312, 76.2673, 0.70, 38)],
        "Ladakh (UT)": [("Leh Town", "Leh", 34.1526, 77.5771, 0.25, 22)],
        "Lakshadweep (UT)": [("Kavaratti Island", "Lakshadweep", 10.5667, 72.6417, 0.15, 18)],
        "Madhya Pradesh": [("Bhopal City", "Bhopal", 23.2599, 77.4126, 1.05, 60)],
        "Maharashtra": [("Mumbai South", "Mumbai City", 18.9388, 72.8353, 1.50, 62), ("Pune Camp", "Pune", 18.5204, 73.8567, 1.00, 46)],
        "Manipur": [("Imphal West", "Imphal West", 24.8170, 93.9368, 0.50, 48)],
        "Meghalaya": [("Shillong Central", "East Khasi Hills", 25.5788, 91.8933, 0.45, 35)],
        "Mizoram": [("Aizawl Town", "Aizawl", 23.7271, 92.7176, 0.38, 30)],
        "Nagaland": [("Kohima South", "Kohima", 25.6751, 94.1086, 0.40, 33)],
        "Odisha": [("Bhubaneswar Central", "Khordha", 20.2961, 85.8245, 0.82, 53)],
        "Puducherry (UT)": [("Puducherry Beach Road", "Puducherry", 11.9416, 79.8083, 0.50, 37)],
        "Punjab": [("Ludhiana North", "Ludhiana", 30.9010, 75.8573, 0.90, 51)],
        "Rajasthan": [("Jaipur Pink City", "Jaipur", 26.9124, 75.7873, 1.08, 56)],
        "Sikkim": [("Gangtok Market", "Gangtok", 27.3389, 88.6065, 0.30, 25)],
        "Tamil Nadu": [("Chennai Central", "Chennai", 13.0827, 80.2707, 1.20, 52)],
        "Telangana": [("Banjara Hills", "Hyderabad", 17.4156, 78.4347, 1.55, 78), ("Kukatpally", "Medchal–Malkajgiri", 17.4948, 78.3996, 1.20, 62)],
        "Tripura": [("Agartala Town", "West Tripura", 23.8315, 91.2868, 0.48, 42)],
        "Uttar Pradesh": [("Lucknow Hazratganj", "Lucknow", 26.8467, 80.9462, 1.45, 68)],
        "Uttarakhand": [("Dehradun Clock Tower", "Dehradun", 30.3165, 78.0322, 0.58, 36)],
        "West Bengal": [("Kolkata Park Street", "Kolkata", 22.5726, 88.3639, 1.30, 65)],
    }
    types = ["Theft", "Assault", "Burglary", "Robbery", "Vandalism", "Fraud"]
    weights = [0.32, 0.16, 0.17, 0.09, 0.17, 0.09]
    rows = []
    for day in pd.date_range(date.today() - timedelta(days=365), periods=365):
        seasonal = 1 + 0.18 * np.sin(2 * np.pi * day.dayofyear / 365)
        weekend = 1.18 if day.dayofweek >= 5 else 1.0
        for state, areas_list in all_states_areas.items():
            for area, district, lat, lon, rate, deprivation in areas_list:
                for _ in range(rng.poisson(rate * seasonal * weekend)):
                    crime = rng.choice(types, p=weights)
                    hour = int(np.clip(rng.normal(18 if crime in ["Theft", "Robbery"] else 13, 4.8), 0, 23))
                    night = int(hour < 6 or hour >= 20)
                    weapon = int(crime in ["Assault", "Robbery"] and rng.random() < 0.27)
                    severity = int(np.clip(round(rng.normal(2.1 + weapon + (0.5 if crime == "Robbery" else 0), 0.7)), 1, 5))
                    response = max(2, rng.normal(10 + deprivation * 0.09 + night * 3 + weapon * 4, 3))
                    rows.append([
                        day, state, district, area, crime, hour, lat + rng.normal(0, 0.006), lon + rng.normal(0, 0.006),
                        day.day_name(), day.month, night, weapon, severity, round(response, 1), deprivation, rng.choice(["Arrest", "Open", "Closed"], p=[0.28, 0.24, 0.48])
                    ])
    return pd.DataFrame(rows, columns=["date", "state", "district", "area", "crime_type", "hour", "latitude", "longitude", "day_name", "month", "night_incident", "weapon_involved", "severity_score", "response_minutes", "deprivation_index", "case_status"])

def model_data(df):
    dates = pd.date_range(df.date.min().normalize(), df.date.max().normalize())
    out = pd.MultiIndex.from_product([df.area.unique(), dates], names=["area", "date"]).to_frame(index=False)
    counts = df.groupby(["area", df.date.dt.normalize()]).size().rename("incidents").reset_index()
    out = out.merge(counts, on=["area", "date"], how="left").fillna({"incidents": 0})
    out["dow"] = out.date.dt.dayofweek
    out["month"] = out.date.dt.month
    out["day_index"] = (out.date - out.date.min()).dt.days
    out["lag_1"] = out.groupby("area").incidents.shift().fillna(0)
    out["rolling_7"] = out.groupby("area").incidents.transform(lambda s: s.shift().rolling(7, min_periods=1).mean()).fillna(0)
    return out

def p_text(p): return f"p = {p:.4f}" + (" (statistically significant at 5%)" if p < .05 else " (not statistically significant at 5%)")

st.markdown("""<style>
.stApp{background:#f4f6f8;color:#182b3a}.hero{background:#1f4e79;padding:1.4rem 1.7rem;border-radius:8px;color:#fff;margin-bottom:1rem;border-left:7px solid #f2b134}.hero h1{margin:0;color:#fff!important;font-size:2rem;font-weight:700}.hero p{margin:.4rem 0 0;color:#fff!important;font-size:1rem}
h1,h2,h3{color:#173f5f!important;font-weight:700!important}p,li,label,.stCaption{color:#263d4d!important}div[data-testid='stMetric']{background:#fff;border:1px solid #b8c7d1;border-radius:6px;padding:13px}div[data-testid='stMetricLabel']{color:#34566f!important;font-weight:700!important;font-size:.9rem!important}div[data-testid='stMetricValue']{color:#173f5f!important;font-weight:800!important;font-size:1.8rem!important}.stTabs [data-baseweb='tab']{color:#263d4d!important;font-size:1rem;font-weight:700}.stTabs [aria-selected='true']{color:#0b5cab!important}.note{background:#fff7df;border-left:4px solid #b7791f;padding:.8rem;border-radius:4px;color:#553c0b!important}div[data-baseweb='select'] span{color:#102a43!important;font-weight:600}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Crime Hotspot Project")
    st.caption("Student Major Project • Python & Streamlit")

df = make_demo_data()

india_directory, directory_loaded = india_state_district_directory()
for state, district in df[["state", "district"]].drop_duplicates().itertuples(index=False):
    india_directory.setdefault(str(state), [])
    if str(district) not in india_directory[str(state)]:
        india_directory[str(state)].append(str(district))

with st.sidebar:
    period = st.date_input("Analysis period", (df.date.min().date(), df.date.max().date()), min_value=df.date.min().date(), max_value=df.date.max().date())
    states = st.multiselect("States / UTs", sorted(df.state.unique()), default=sorted(df.state.unique()))
    districts = st.multiselect("Districts", sorted(df[df.state.isin(states)].district.unique()), default=sorted(df[df.state.isin(states)].district.unique()))
    available_areas = sorted(df[(df.state.isin(states)) & (df.district.isin(districts))].area.unique())
    selected_areas = st.multiselect("Areas / police divisions", available_areas, default=available_areas)
    types = st.multiselect("Crime categories", sorted(df.crime_type.unique()), default=sorted(df.crime_type.unique()))

if len(period) != 2:
    st.info("Choose a start and end date.")
    st.stop()

f = df[df.date.dt.date.between(*period) & df.state.isin(states) & df.district.isin(districts) & df.area.isin(selected_areas) & df.crime_type.isin(types)].copy()
if f.empty:
    st.warning("No records match the applied filters.")
    st.stop()

st.markdown("""<div class='hero'><h1>Crime Hotspot Prediction System</h1><p>Student project using data science, statistical analysis and machine learning.</p></div>""", unsafe_allow_html=True)
st.caption("This dashboard is designed for academic demonstration. It analyses aggregate incident data and does not identify individuals.")
days = max((pd.Timestamp(period[1]) - pd.Timestamp(period[0])).days + 1, 1)
recent = f[f.date >= f.date.max() - pd.Timedelta(days=30)]
prior = f[(f.date < f.date.max() - pd.Timedelta(days=30)) & (f.date >= f.date.max() - pd.Timedelta(days=60))]
trend = 0 if prior.empty else (len(recent) - len(prior)) / len(prior) * 100

a, b, c, d = st.columns(4)
a.metric("Study observations", f"{len(f):,}")
b.metric("States / UTs", str(f.state.nunique()))
c.metric("Districts", str(f.district.nunique()))
d.metric("30-day trend", f"{trend:+.1f}%")

overview, methods, forecast, quality, explorer = st.tabs(["Dashboard", "Statistical analysis", "Hotspot prediction", "Study design & quality", "Data explorer"])
with overview:
    l, r = st.columns([1.25, 1])
    daily = f.groupby(f.date.dt.date).size().rename("Incidents").reset_index().rename(columns={"date": "Date"})
    with l: st.plotly_chart(px.area(daily, x="Date", y="Incidents", title="Incident volume over time", color_discrete_sequence=["#246eb9"]), use_container_width=True)
    with r:
        ac = f.area.value_counts().rename_axis("Area").reset_index(name="Incidents").sort_values("Incidents")
        st.plotly_chart(px.bar(ac, x="Incidents", y="Area", orientation="h", title="Incident concentration by area", color="Incidents", color_continuous_scale="Blues"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(px.pie(f.crime_type.value_counts().reset_index(name="Incidents"), names="crime_type", values="Incidents", hole=.5, title="Crime composition"), use_container_width=True)
    with c2: st.plotly_chart(px.bar(f.groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="Incidents"), x="hour", y="Incidents", title="Incidents by hour", labels={"hour": "Hour of day"}, color_discrete_sequence=["#1d4e89"]), use_container_width=True)
    if {"latitude", "longitude"}.issubset(f) and f["latitude"].notna().any():
        st.subheader("Geographic distribution")
        st.map(f.dropna(subset=["latitude", "longitude"]).rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

with methods:
    st.subheader("Core Statistical Analysis")
    st.caption("Essential statistical evaluations applied to the filtered dataset.")
    
    # 1. Spatial Independence
    ct = pd.crosstab(f.area, f.crime_type)
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    st.markdown(f"**1. Spatial Chi-Square Test** — H₀: Crime types are distributed independent of location. χ² = {chi2:.2f}, df = {dof}, {p_text(p_chi)}.")
    
    # 2. Predictive Outcome Modeling
    x = pd.get_dummies(f[["hour", "severity_score", "deprivation_index", "night_incident"]], drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = (f.case_status == "Arrest").astype(int)
    if y.nunique() > 1:
        lr = LogisticRegression(max_iter=1000).fit(x, y)
        prob = lr.predict_proba(x)[:, 1]
        st.markdown(f"**2. Logistic Regression (Arrest Determinants)** — In-sample accuracy = {accuracy_score(y, prob >= .5):.2%}; ROC-AUC = {roc_auc_score(y, prob):.2f}. Measures the strength of association between timing/severity features and successful case resolution.")
    
    # 3. Aggregated Profiling
    st.markdown("**3. Descriptive & Concentration Analysis** — Incident frequency, regional density, and peak hours are mapped dynamically on the Dashboard.")

with forecast:
    st.subheader("Next-day area hotspot prediction")
    st.caption("A Random Forest model predicts the possible number of incidents. It uses day, month, previous-day incidents, and the previous 7-day average.")
    md = model_data(f)
    feats = ["dow", "month", "day_index", "lag_1", "rolling_7"]
    split = max(14, len(md) // 8)
    train, test = md.iloc[:-split], md.iloc[-split:]
    if len(train) > 30 and train.incidents.nunique() > 1:
        m = RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=1).fit(train[feats], train.incidents)
        pred = m.predict(test[feats])
        x, y = st.columns(2)
        x.metric("Average prediction error (MAE)", f"{mean_absolute_error(test.incidents, pred):.2f} incidents/day")
        y.metric("Model score (R²)", f"{r2_score(test.incidents, pred):.2f}")
        n = md.sort_values("date").groupby("area").tail(1).copy()
        n.date += pd.Timedelta(days=1)
        n.dow = n.date.dt.dayofweek
        n.month = n.date.dt.month
        n.day_index += 1
        n.lag_1 = n.incidents
        n["Predicted incidents"] = m.predict(n[feats]).clip(0)
        n["Risk level"] = pd.cut(n["Predicted incidents"], [-1, 1.2, 2.5, np.inf], labels=["Low", "Medium", "High"])
        result = n[["area", "Predicted incidents", "Risk level", "rolling_7"]].rename(columns={"area": "Area", "rolling_7": "7-day average"}).sort_values("Predicted incidents", ascending=False)
        st.dataframe(result.style.format({"Predicted incidents": "{:.2f}", "7-day average": "{:.2f}"}).background_gradient(subset=["Predicted incidents"], cmap="YlOrRd"), use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(result, x="Area", y="Predicted incidents", color="Risk level", title="Prioritised areas for tomorrow", color_discrete_map={"High": "#d64b4b", "Medium": "#dc8b22", "Low": "#25805e"}), use_container_width=True)
    else:
        st.info("More variation and at least 45 area-days are needed for forecasting.")

with quality:
    st.subheader("Research design, ethics, and data quality")
    st.markdown("**Objectives:** (1) describe crime patterns; (2) test whether patterns differ by location, timing, and type; (3) estimate risk factors; (4) forecast area-level incident volume; and (5) support fair, preventive resource allocation.")
    st.markdown("**Measurement scales:** nominal (area, crime type, case status), binary (night/weapon), ordinal (severity score), interval/ratio (hour, response minutes, deprivation index, coordinates), and temporal (date).")
    q = pd.DataFrame({"Variable": f.columns, "Data type": [str(f[c].dtype) for c in f], "Missing values": [int(f[c].isna().sum()) for c in f], "Unique values": [int(f[c].nunique()) for c in f]})
    st.dataframe(q, use_container_width=True, hide_index=True)
    st.markdown("<div class='note'><b>Ethical limitation:</b> hotspot signals can reflect reporting and enforcement patterns. Do not use them to target individuals or protected groups. Use aggregate areas, review bias, protect privacy, and combine the model with local expertise.</div>", unsafe_allow_html=True)

with explorer:
    st.subheader("Filtered research dataset")
    st.dataframe(f.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download filtered data", f.to_csv(index=False).encode(), "crime_incidents_filtered.csv", "text/csv")