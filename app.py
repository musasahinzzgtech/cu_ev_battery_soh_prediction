import os
import glob

import joblib
import pandas as pd
import streamlit as st

import config
import preprocessing

modelPath = os.path.join(config.modelsDir, "model.joblib")
prepPath = os.path.join(config.modelsDir, "prep.joblib")


@st.cache_resource
def loadModelAndPrep():
    model = joblib.load(modelPath)
    prep = joblib.load(prepPath)
    return model, prep


@st.cache_data
def loadDataset():
    return pd.read_csv(config.dataPath)


def predictSoh(model, prep, inputs):
    row = pd.DataFrame([inputs])
    X = preprocessing.makeFeatures(row)
    X = X.reindex(columns=prep["features"], fill_value=0.0)
    X[config.numericFeatures] = prep["scaler"].transform(X[config.numericFeatures])
    return float(model.predict(X)[0])


def showImages(specs):
    shown = 0
    for filename, caption in specs:
        path = os.path.join(config.figuresDir, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
            shown += 1
    if shown == 0:
        st.info("No figures yet. Run `python main.py` to create them.")


st.set_page_config(page_title="EV Battery SoH Predictor", page_icon="🔋", layout="wide")
st.title("EV Battery State-of-Health Predictor")
st.write(
    "Enter the battery info below to estimate its State of Health (SoH). "
    "This is the demo for the thesis project."
)

df = loadDataset()

if not os.path.exists(modelPath) or not os.path.exists(prepPath):
    st.error("Model not found. Please run `python main.py` first to train it.")
    st.stop()

model, prep = loadModelAndPrep()

tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "Model Performance"])

with tab1:
    st.subheader("Battery info")

    carModels = sorted(df["Car_Model"].dropna().unique().tolist())
    batteryTypes = sorted(df["Battery_Type"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    with c1:
        carModel = st.selectbox("Car model", carModels)
    with c2:
        batteryType = st.selectbox("Battery type", batteryTypes)
    with c3:
        drivingStyle = st.selectbox("Driving style", list(config.ordinalMap))

    labels = {
        "Battery_Capacity_kWh": "Battery capacity (kWh)",
        "Vehicle_Age_Months": "Vehicle age (months)",
        "Total_Charging_Cycles": "Total charging cycles",
        "Avg_Temperature_C": "Average temperature (°C)",
        "Fast_Charge_Ratio": "Fast-charge ratio",
        "Avg_Discharge_Rate_C": "Average discharge rate (C)",
        "Internal_Resistance_Ohm": "Internal resistance (Ohm)",
    }

    numbers = {}
    cols = st.columns(2)
    i = 0
    for feature in config.numericFeatures:
        colMin = float(df[feature].min())
        colMax = float(df[feature].max())
        colMed = float(df[feature].median())
        with cols[i % 2]:
            numbers[feature] = st.slider(labels[feature], colMin, colMax, colMed)
        i += 1

    if st.button("Predict SoH", type="primary"):
        inputs = {"Car_Model": carModel, "Battery_Type": batteryType,
                  config.ordinalCol: drivingStyle}
        inputs.update(numbers)
        soh = predictSoh(model, prep, inputs)
        soh = max(0.0, min(100.0, soh))

        st.metric("Predicted State of Health", str(round(soh, 1)) + "%")
        if soh >= 90:
            st.success("Healthy - the battery is in good condition.")
        elif soh >= 80:
            st.warning("Caution - some degradation, keep an eye on it.")
        else:
            st.error("Critical - a lot of degradation, needs maintenance.")

with tab2:
    st.subheader("Data charts")
    showImages([
        ("feature_distributions.png", "Feature distributions"),
        ("soh_boxplot_by_chemistry.png", "SoH by battery type"),
        ("temperature_vs_soh.png", "Temperature vs SoH"),
        ("correlation_heatmap.png", "Correlation heatmap"),
        ("cycles_vs_soh.png", "Charging cycles vs SoH"),
        ("arrhenius_degradation.png", "Arrhenius plot"),
    ])

with tab3:
    st.subheader("Model performance")
    found = 0
    for pattern, heading in [("*_feature_importance.png", "Feature importance"),
                             ("*_pred_vs_actual.png", "Predicted vs actual"),
                             ("*_residuals.png", "Residuals")]:
        matches = sorted(glob.glob(os.path.join(config.figuresDir, pattern)))
        if matches:
            st.markdown("**" + heading + "**")
            for path in matches:
                st.image(path, caption=os.path.basename(path), use_container_width=True)
                found += 1
    if found == 0:
        st.info("No performance charts yet. Run `python main.py` first.")
