import os
import glob

import joblib
import pandas as pd
import streamlit as st

import config
import preprocessing

LITERATURE = [
    {
        "id": 1,
        "title": "State of Health Estimation in EV Batteries Using AI-enhanced BMS",
        "authors": "Obisike, K. C., Kukuchuku, S., Abidde, W. N., & Ibanibo, T. S.",
        "year": 2025,
        "journal": "International Journal of Artificial Intelligence of Things (AIoT) in Communication Industry, 1(2), 6–14",
        "url": "https://www.researchgate.net/publication/396179390_State_of_Health_Estimation_in_EV_Batteries_Using_AI-_enhanced_BMS",
        "reference": "Obisike, K. C., Kukuchuku, S., Abidde, W. N., ve Ibanibo, T. S. (2025). State of health estimation in EV batteries using AI-enhanced BMS. International Journal of Artificial Intelligence of Things (AIoT) in Communication Industry, 1(2), 6–14.",
        "type": "Empirical Study",
        "tags": ["LSTM", "XGBoost", "Hybrid AI", "BMS", "PCA", "STFT"],
        "overview": (
            "Presents an AI-enhanced Battery Management System (BMS) framework using a hybrid ML pipeline "
            "to predict SoH of lithium-ion batteries in EVs. Shifts away from traditional model-based approaches "
            "to capture complex, non-linear degradation patterns over time."
        ),
        "aim": (
            "Enhance safety, reliability, and lifespan of EV battery systems by accurately estimating SoH in "
            "real-time using a hybrid LSTM + Gradient Boosting model trained on publicly available lab datasets."
        ),
        "methods": (
            "**LSTM:** Models temporal degradation trends over 30-cycle windows.\n\n"
            "**XGBoost (Gradient Boosting):** Processes LSTM sequence outputs alongside auxiliary features for final prediction.\n\n"
            "**PCA:** Dimensionality reduction retaining 95% variance.\n\n"
            "**STFT:** Extracts frequency-domain features from battery operational data.\n\n"
            "**Monte Carlo Dropout:** Provides uncertainty estimates and probabilistic diagnostics.\n\n"
            "Data: NASA Ames and CALCE datasets (LiCoO₂, NMC). Split: 70/15/15."
        ),
        "results": (
            "| Metric | Finding |\n|---|---|\n"
            "| Prediction Accuracy | >97% on test datasets |\n"
            "| Average RMSE | <1.5% |\n"
            "| Stability Improvement | 20–25% over conventional algorithms |\n"
            "| Error (GB model) | Std dev ~0.9% vs LSTM ~1.36% |"
        ),
        "conclusion": (
            "Integrating hybrid AI (LSTM + GB) into a BMS reliably smooths noise, accurately predicts end-of-life, "
            "and provides stable SoH estimations practical for real-world automotive deployment."
        ),
        "pdf": "predicting_ev_battery_literature_sample_1.pdf",
    },
    {
        "id": 2,
        "title": "Predictive Modeling for Electric Vehicle Battery State of Health: A Comprehensive Literature Review",
        "authors": "Gong, J., Xu, B., Chen, F., & Zhou, G.",
        "year": 2025,
        "journal": "Energies, 18(2), 337",
        "url": "https://www.mdpi.com/1996-1073/18/2/337",
        "reference": "Gong, J., Xu, B., Chen, F., ve Zhou, G. (2025). Predictive modeling for electric vehicle battery state of health: A comprehensive literature review. Energies, 18(2), 337. https://doi.org/10.3390/en18020337",
        "type": "Literature Review",
        "tags": ["SVM", "LSTM", "GPR", "CatBoost", "Kalman Filter", "ECM", "Hybrid Models"],
        "overview": (
            "A comprehensive synthesis of SOH estimation research from 2014–2024, evaluating direct, physics-based, "
            "data-driven, and hybrid approaches. Covers key health indicators and real-world degradation factors "
            "often overlooked in lab settings."
        ),
        "aim": (
            "Provide a unified, exhaustive comparison of all diverse SOH estimation approaches and systematically "
            "review health indicators, degradation factors, and required datasets."
        ),
        "methods": (
            "**Direct Measurement:** Coulomb Counting, EIS — simple but error-prone over time.\n\n"
            "**Physics-Based:** ECM (Thevenin, Randle's), DFN model, Kalman Filters — interpretable but computationally heavy.\n\n"
            "**Data-Driven:** SVM, ANN, RNN, LSTM, DCNN, GPR, CatBoost — high accuracy, black-box.\n\n"
            "**Hybrid:** Fusing LSTM/GPR with physics-based ECM/Particle Filters — best of both worlds.\n\n"
            "Key SOH indicators: capacity fading, internal resistance, CCDT, MCCDR."
        ),
        "results": (
            "**Critical issue:** Most models trained on offline lab data fail to represent real-world EV conditions.\n\n"
            "**Future direction:** Hybrid models combining AI speed/accuracy with physical interpretability.\n\n"
            "**Next frontier:** Whole battery *pack* SOH estimation (not just single cells), accounting for cell-to-cell variability."
        ),
        "conclusion": (
            "No single method is perfect. The future of SOH estimation lies in hybrid models that leverage AI "
            "for pattern recognition while grounding results in physical battery mechanics."
        ),
        "pdf": "predicting_ev_battery_literature_sample_2.pdf",
    },
    {
        "id": 3,
        "title": "Predicting EV Battery State of Health Using Long Short-Term Degradation Feature Extraction and FEA-TimeMixer",
        "authors": "Tang, W., Chen, J., & Chen, D.",
        "year": 2025,
        "journal": "Scientific Reports, 15, 2200",
        "url": "https://www.nature.com/articles/s41598-025-85492-3",
        "reference": "Tang, W., Chen, J., ve Chen, D. (2025). Predicting EV battery state of health using long short term degradation feature extraction and FEA TimeMixer. Scientific Reports, 15, 2200. https://doi.org/10.1038/s41598-025-85492-3",
        "type": "Novel Architecture",
        "tags": ["FEA-TimeMixer", "CEEMDAN", "Autoencoder", "Fourier Transform", "Savitzky-Golay"],
        "overview": (
            "Proposes a novel hybrid architecture called FEA-TimeMixer for SOH prediction using real-world taxi "
            "charging data. Addresses capacity recovery phenomena and error accumulation in long-term forecasting "
            "by combining empirical degradation models with frequency-domain deep learning."
        ),
        "aim": (
            "Improve SOH prediction accuracy across short and long prediction horizons using real-world operational "
            "data, overcoming limitations of existing deep learning models that struggle with frequency-domain features."
        ),
        "methods": (
            "**SOH Label Algorithm:** Sliding window with Savitzky-Golay smoothing + PCHIP interpolation.\n\n"
            "**Empirical Models:** Linear, Lasso, ElasticNet, Exponential, Double Exponential (best: Exponential).\n\n"
            "**CEEMDAN:** Decomposes capacity data into frequency components, isolating noise from long-term trends.\n\n"
            "**Autoencoder:** Fuses long-term + short-term features into a compressed 1×10 vector.\n\n"
            "**FEA (Frequency Enhanced Attention):** Fourier-based attention for spectral information.\n\n"
            "**TimeMixer:** Multiscale forecasting with PDM + FMM blocks.\n\n"
            "Data: 6 real-world EV taxis, ~2 years, CAN bus at 8s intervals. Split: 70/10/20."
        ),
        "results": (
            "| Horizon | MAE |\n|---|---|\n"
            "| Short-term | <0.0219 |\n"
            "| Long-term | <0.1007 |\n\n"
            "Temperature finding: high temps (~40°C) caused capacity recovery; low temps (~25°C) caused rapid decline. "
            "FEA block outperformed base TimeMixer, proving spectral features boost precision."
        ),
        "conclusion": (
            "Combining empirical degradation models with CEEMDAN and FEA-TimeMixer creates a robust framework "
            "that handles noisy real-world data and excels in both short- and long-term SOH forecasting."
        ),
        "pdf": "predicting_ev_battery_literature_sample_3.pdf",
    },
    {
        "id": 4,
        "title": "SOH Illusion: Misunderstandings of EV Battery State of Health and Methods to Promote Understanding",
        "authors": "Lin, K. R., Li, J., Sparks, J., Filipowicz, A. L. S., Shamma, D. A., & Libby, L. A.",
        "year": 2025,
        "journal": "AutomotiveUI '25, 17th Intl. Conference on Automotive User Interfaces, 129–138",
        "url": "https://dl.acm.org/doi/10.1145/3744333.3747828",
        "reference": "Lin, K. R., Li, J., Sparks, J., Filipowicz, A. L. S., Shamma, D. A., ve Libby, L. A. (2025). SOH Illusion: Misunderstandings of EV Battery State of Health and Methods to Promote Understanding. AutomotiveUI '25, 129–138. https://doi.org/10.1145/3744333.3747828",
        "type": "HCI Study",
        "tags": ["UI Design", "Cognitive Bias", "Dashboard", "Behavioral Science", "ANOVA", "Survey Study"],
        "overview": (
            "A Human-Computer Interaction study investigating how everyday drivers interpret SOH displays. "
            "With new legislation requiring EVs to show SOH on dashboards, the research tests whether users "
            "understand non-linear battery degradation and how UI design can correct dangerous misconceptions."
        ),
        "aim": (
            "Determine whether EV owners understand SOH and battery degradation's non-linear nature (the 'SOH Illusion'), "
            "and test whether different visual dashboard designs can correct these misconceptions."
        ),
        "methods": (
            "**Study 1:** 558 participants surveyed on SOH vs SOC confusion; asked to predict battery health across 3 life stages.\n\n"
            "**Study 2:** 1,995 participants tested across 5 UI frames: text percentage, gauge chart, bullet chart, gauge+legend, bullet+legend.\n\n"
            "**Statistical Methods:** Logistic Mixed Effects models, Type III ANOVA, GLHT with Tukey's HSD."
        ),
        "results": (
            "| Finding | Detail |\n|---|---|\n"
            "| SOH Illusion confirmed | ~46% falsely expected linear degradation |\n"
            "| SOH vs Usable Life | 1/3 incorrectly assumed 90% SOH = 90% life remaining |\n"
            "| UI Visualizations | Reduced linear assumptions by 2×–4.4× |\n"
            "| Legends | No additional benefit over color-coded charts alone |\n"
            "| Late-Stage | Users still struggled to predict the sudden 'end of life' drop |"
        ),
        "conclusion": (
            "Thoughtfully designed visual interfaces (color-coded gauge or bullet charts) can subconsciously "
            "correct the cognitive bias of linear degradation expectations, leading to better-informed drivers."
        ),
        "pdf": "predicting_ev_battery_literature_sample_4.pdf",
    },
    {
        "id": 5,
        "title": "Estimation of State of Health for Lithium-Ion Batteries Using Advanced Data-Driven Techniques",
        "authors": "Rout, S., Samal, S. K., Gelmecha, D. J., & Mishra, S.",
        "year": 2025,
        "journal": "Scientific Reports, 15, 30438",
        "url": "https://www.nature.com/articles/s41598-025-93775-y",
        "reference": "Rout, S., Samal, S. K., Gelmecha, D. J., ve Mishra, S. (2025). Estimation of state of health for lithium-ion batteries using advanced data-driven techniques. Scientific Reports, 15, 30438. https://doi.org/10.1038/s41598-025-93775-y",
        "type": "Empirical Study",
        "tags": ["LSTM", "ANN", "Random Forest", "XGBoost", "AdaBoost", "RUL Prediction", "LEV"],
        "overview": (
            "Presents a comprehensive ML framework for estimating SOH and predicting Remaining Useful Life (RUL) "
            "of lithium-ion batteries in lightweight three-wheeled electric vehicles (LEVs). Compares seven ML "
            "algorithms on real CAN bus data from a lab-designed EV prototype."
        ),
        "aim": (
            "Accurately model nonlinear Li-ion battery degradation to estimate current health and predict future "
            "lifespan, with algorithms efficient enough for real-time BMS deployment."
        ),
        "methods": (
            "**Data:** 66,302 data points from NMC 18650 cells via CAN bus BMS. Features: cell voltage, current, "
            "internal resistance, temperature. Split: 80/20.\n\n"
            "**Models compared:** LSTM (winner), ANN, Random Forest, XGBoost, AdaBoost, Decision Tree, Ridge Regression.\n\n"
            "**LSTM config:** Input → 2×(64 neurons, ReLU) hidden layers → linear output. Adam optimizer, MSE loss, "
            "Dropout 0.2, Early Stopping.\n\n"
            "**Preprocessing:** Min-Max normalization."
        ),
        "results": (
            "| Model | R² | MSE |\n|---|---|---|\n"
            "| LSTM | 0.9982 | 0.000115 |\n"
            "| ANN | 0.9890 | — |\n"
            "| Random Forest | 0.9839 | — |\n"
            "| XGBoost | 0.9402 | — |\n\n"
            "**RUL prediction:** LSTM predicted 9.7-year remaining battery life (MAE: 0.065)."
        ),
        "conclusion": (
            "LSTM networks are highly effective for battery SOH and RUL prediction due to their ability to "
            "capture dynamic, nonlinear decay. Integrating these into a LEV's BMS ensures safer operation and "
            "reliable driver dashboards."
        ),
        "pdf": "predicting_ev_battery_literature_sample_5.pdf",
    },
    {
        "id": 6,
        "title": "A Review of Improvements on Electric Vehicle Battery",
        "authors": "Koech, A. K., Mwandila, G., & Mulolani, F.",
        "year": 2024,
        "journal": "Heliyon, 10, e34806",
        "url": "https://www.sciencedirect.com/science/article/pii/S2405844024108377",
        "reference": "Koech, A. K., Mwandila, G., ve Mulolani, F. (2024). A review of improvements on electric vehicle battery. Heliyon, 10, e34806. https://doi.org/10.1016/j.heliyon.2024.e34806",
        "type": "Literature Review",
        "tags": ["Li-ion Chemistry", "Solid-State", "SEI", "NMC", "LFP", "Anode", "Cathode"],
        "overview": (
            "A comprehensive review of advancements in EV battery chemistry and architecture. Covers improvements "
            "to the four main battery components — anode, cathode, electrolyte, and separator — and addresses "
            "critical safety challenges like lithium plating and dendrite formation."
        ),
        "aim": (
            "Outline the chemical and structural innovations needed to achieve longer range, faster charging, and "
            "improved safety in EV batteries, projecting market growth from 10.5M units (2022) to 74.5M by 2035."
        ),
        "methods": (
            "**Anode:** Shift from graphite to silicon/tin composites, graphene/MXenes, and metallic lithium with 3D carbon matrices.\n\n"
            "**Cathode:** From LCO to NMC (industry favorite) toward cobalt-free NMO alternatives with surface coatings.\n\n"
            "**Electrolyte:** Fluorinated solvents (up to 5.6V stability) and Solid-State Electrolytes (SSE) for safety.\n\n"
            "**SEI (Solid Electrolyte Interphase):** Electrolyte additives (FEC, VC) for self-repairing artificial layers.\n\n"
            "**Separator:** Ceramic-coated polymers, electrospun nanofibers — heat-stable up to 200°C."
        ),
        "results": (
            "**Lithium plating & dendrites:** Major safety risk during fast charging and cold temperatures. "
            "Mitigated via optimal N/P ratio, battery pre-heating, and dynamic BMS throttling.\n\n"
            "**Solid-state batteries:** Remain in lab-phase but promise transformative safety and energy density.\n\n"
            "**Cobalt-free cathodes:** Critical for cost and ethical sourcing, requiring surface coating advances."
        ),
        "conclusion": (
            "Future breakthroughs rely on moving solid-state batteries out of the lab, designing cobalt-free cathodes, "
            "and mastering atomic-level interactions at the SEI."
        ),
        "pdf": "predicting_ev_battery_literature_sample_6.pdf",
    },
    {
        "id": 7,
        "title": "Integrating Battery Aging in the Optimization for Bidirectional Charging of Electric Vehicles",
        "authors": "Naveena, S., Kalyan, S. B., Jayaprakash, K., Prasanth, S., & Mahandhraselvan, T.",
        "year": 2026,
        "journal": "International Journal of Creative and Open Research in Engineering and Management, 2(3), 1–9",
        "url": "https://www.researchgate.net/publication/403400277_Integrating_Battery_Aging_in_the_Optimization_for_Bidirectional_Charging_of_Electric_Vehicles",
        "reference": "Naveena, S., Kalyan, S. B., Jayaprakash, K., Prasanth, S., ve Mahandhraselvan, T. (2026). Integrating battery aging in the optimization for bidirectional charging of electric vehicles. International Journal of Creative and Open Research in Engineering and Management, 2(3), 1–9. https://doi.org/10.55041/ijcope.v2i3.248",
        "type": "Optimization Study",
        "tags": ["V2G", "MILP", "NLP", "Cycle Aging", "Calendar Aging", "Smart Grid"],
        "overview": (
            "Explores Vehicle-to-Grid (V2G) bidirectional charging in smart grids. Proposes an optimization "
            "framework balancing financial profits from energy trading against long-term battery wear costs, "
            "using a fleet of 50 EVs in a residential grid simulation."
        ),
        "aim": (
            "Develop an aging-aware optimization strategy for V2G participation that protects battery lifespan "
            "while enabling profitable grid energy trading."
        ),
        "methods": (
            "**Cycle Aging Model:** Degradation from active use — influenced by DoD, C-rate, temperature.\n\n"
            "**Calendar Aging Model:** Degradation at rest — influenced by average SoC and ambient temperature.\n\n"
            "**Optimization:** MILP + NLP to minimize: (Grid electricity cost) − (V2G revenue) + (Battery degradation cost).\n\n"
            "**Constraints:** SoC bounds (20%–90%), 7 kW power limits, departure SoC targets.\n\n"
            "**Simulation:** 50 EVs × 40 kWh batteries. Three scenarios tested."
        ),
        "results": (
            "| Scenario | Outcome |\n|---|---|\n"
            "| Baseline (Unidirectional) | No grid support, baseline degradation |\n"
            "| V2G Without Aging | +30–40% degradation — financially unsustainable |\n"
            "| V2G With Aging (Proposed) | 20–25% degradation reduction vs. aggressive model |\n\n"
            "Battery penalty: ₹5,00,000 replacement cost / 3,000 cycles used to teach the algorithm moderation."
        ),
        "conclusion": (
            "Aging-aware V2G optimization successfully balances profit and battery health. Future work should "
            "integrate Reinforcement Learning for dynamic degradation prediction based on historical driving data."
        ),
        "pdf": "predicting_ev_battery_literature_sample_7.pdf",
    },
]

TYPE_COLORS = {
    "Empirical Study": "#1f77b4",
    "Literature Review": "#2ca02c",
    "Novel Architecture": "#9467bd",
    "HCI Study": "#d62728",
    "Optimization Study": "#17becf",
}


MODEL_TITLE_MAP = {
    "linearregression": "Linear Regression",
    "randomforest": "Random Forest",
    "elasticnet": "Elastic Net",
    "extratrees": "Extra Trees",
    "svr": "SVR",
    "histgradientboosting": "Hist Gradient Boosting",
    "xgboost": "XGBoost",
    "catboostregressor": "CatBoost",
}

MODEL_DESCRIPTIONS = {
    "catboostregressor": (
        "CatBoost (Categorical Boosting) is a gradient-boosting algorithm that uses *ordered boosting* "
        "and native categorical feature handling to reduce prediction shift. It ranked first on all three "
        "test-set metrics in this thesis — R² 0.9893, RMSE 0.3370, MAE 0.2604 — narrowly ahead of "
        "XGBoost and SVR. The small gap between the top three (< 0.01 RMSE) indicates a shared "
        "performance ceiling on this dataset."
    ),
    "xgboost": (
        "XGBoost (Extreme Gradient Boosting) is a regularised gradient-boosting library that appeared "
        "as both a standalone and hybrid component in prior SoH literature. It ranked second "
        "(R² 0.9889, RMSE 0.3443, MAE 0.2660), within 0.008 RMSE of CatBoost."
    ),
    "svr": (
        "Support Vector Regression fits within an ε-insensitive margin using a kernel trick. "
        "It ranked third (R² 0.9888, RMSE 0.3450, MAE 0.2642), confirming that kernel methods "
        "stay competitive when features are properly scaled."
    ),
    "extratrees": (
        "Extra Trees (Extremely Randomised Trees) adds random split thresholds to bagging, "
        "lowering variance and speeding up training. It ranked fourth (R² 0.9879, RMSE 0.3580)."
    ),
    "histgradientboosting": (
        "Histogram-based Gradient Boosting bins features for fast boosting. "
        "It ranked fifth (R² 0.9874, RMSE 0.3663)."
    ),
    "randomforest": (
        "Random Forest is a bagging ensemble that averages decorrelated decision trees. "
        "It ranked sixth (R² 0.9867, RMSE 0.3754)."
    ),
    "elasticnet": (
        "Elastic Net adds combined L1+L2 regularisation to linear regression. "
        "It ranked last alongside Linear Regression (R² 0.9839, RMSE 0.4134), confirming "
        "that the linear form — not collinearity — is the bottleneck."
    ),
    "linearregression": (
        "Ordinary Least Squares baseline. Ranked last (R² 0.9839, RMSE 0.4132). "
        "Its near-identical score to Elastic Net confirms collinearity was not the main limit — "
        "linearity itself is."
    ),
}

ALL_METRICS = [
    ("CatBoost",             0.2604, 0.3370, 0.9893),
    ("XGBoost",              0.2660, 0.3443, 0.9889),
    ("SVR",                  0.2642, 0.3450, 0.9888),
    ("Extra Trees",          0.2818, 0.3580, 0.9879),
    ("Hist Gradient Boosting", 0.2758, 0.3663, 0.9874),
    ("Random Forest",        0.2884, 0.3754, 0.9867),
    ("Linear Regression",    0.3110, 0.4132, 0.9839),
    ("Elastic Net",          0.3107, 0.4134, 0.9839),
]


@st.cache_data
def _pdfBytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

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


st.set_page_config(page_title="EV Battery SoH Predictor", page_icon="cukurova_logo.svg", layout="wide")
st.logo("cukurova_logo.svg", link="https://www.cu.edu.tr")
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

if "app_ready" not in st.session_state:
    st.markdown("#### Preparing application…")
    bar = st.progress(0)
    for _i, _paper in enumerate(LITERATURE):
        bar.progress(
            (_i + 1) / len(LITERATURE),
            text=f"Loading: {_paper['title'][:70]}…",
        )
        _pdfPath = os.path.join("literatureDocs", _paper["pdf"])
        if os.path.exists(_pdfPath):
            _pdfBytes(_pdfPath)
    st.session_state["app_ready"] = True
    st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Data Insights", "Model Performance", "Literature"])

with tab1:
    modelKey = type(model).__name__.lower()
    modelTitle = MODEL_TITLE_MAP.get(modelKey, type(model).__name__)
    modelDesc = MODEL_DESCRIPTIONS.get(modelKey, "")

    st.info(
        f"**Active model: {modelTitle}** &nbsp;·&nbsp; "
        "Selected by lowest 5-fold cross-validated RMSE across 8 candidate regressors "
        "(Linear Regression, Elastic Net, Random Forest, Extra Trees, SVR, "
        "Hist Gradient Boosting, XGBoost, CatBoost).",
    )

    with st.expander("Why this model? — Ranking & description"):
        if modelDesc:
            st.markdown(modelDesc)
        st.markdown("**Full test-set results (ordered by RMSE):**")
        metricsData = {
            "Model": [r[0] for r in ALL_METRICS],
            "MAE":   [r[1] for r in ALL_METRICS],
            "RMSE":  [r[2] for r in ALL_METRICS],
            "R²":    [r[3] for r in ALL_METRICS],
        }
        metricsDf = pd.DataFrame(metricsData).set_index("Model")
        st.dataframe(
            metricsDf.style.highlight_min(subset=["MAE", "RMSE"], color="#d4edda")
                           .highlight_max(subset=["R²"], color="#d4edda")
                           .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
            use_container_width=True,
        )
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

    figTypes = [
        ("_pred_vs_actual.png", "Predicted vs actual"),
        ("_residuals.png", "Residuals"),
        ("_feature_importance.png", "Feature importance"),
    ]

    modelSlugsSet = set()
    for suffix, _ in figTypes:
        matched = glob.glob(os.path.join(config.figuresDir, "*" + suffix))
        for path in matched:
            filename = os.path.basename(path)
            if filename.endswith(suffix):
                modelSlugsSet.add(filename[: -len(suffix)])
    modelSlugs = sorted(modelSlugsSet)

    if not modelSlugs:
        st.info("No performance charts yet. Run `python main.py` first.")
    else:
        for slug in modelSlugs:
            title = MODEL_TITLE_MAP.get(slug.lower(), slug.replace("_", " ").title())
            with st.expander(title, expanded=False):
                cols = st.columns(len(figTypes))
                for col, (suffix, heading) in zip(cols, figTypes):
                    path = os.path.join(config.figuresDir, slug + suffix)
                    with col:
                        if os.path.exists(path):
                            st.image(path, caption=heading, use_container_width=True)
                        else:
                            st.caption(heading + ": n/a")

with tab4:
    st.markdown(
        """
        <div id="lit-loader" style="
            display:flex;align-items:center;gap:14px;
            padding:18px 20px;margin-bottom:8px;
            background:#f0f4ff;border-radius:10px;border:1px solid #d0d8f0;
            animation: litFadeOut 0.5s ease-out 2s forwards;
        ">
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
               style="flex-shrink:0;animation:litSpin 0.9s linear infinite;">
            <circle cx="12" cy="12" r="10" stroke="#c5cfe8" stroke-width="3"/>
            <path d="M12 2a10 10 0 0 1 10 10" stroke="#1f77b4" stroke-width="3" stroke-linecap="round"/>
          </svg>
          <div>
            <div style="font-weight:600;color:#1a1a2e;font-size:0.95rem;">
              Loading literature documents…
            </div>
            <div style="color:#555;font-size:0.78rem;margin-top:2px;">
              7 papers · EV Battery State-of-Health Estimation
            </div>
          </div>
        </div>
        <style>
          @keyframes litSpin { to { transform: rotate(360deg); } }
          @keyframes litFadeOut { to { opacity:0; height:0; padding:0; margin:0; border:none; overflow:hidden; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Literature Review")
    st.caption(f"{len(LITERATURE)} papers referenced in this thesis on EV battery State of Health estimation.")

    st.markdown(
        """
        <style>
        .lit-tag {
            display: inline-block;
            background: #f0f2f6;
            color: #444;
            border-radius: 4px;
            padding: 2px 8px;
            font-size: 0.75rem;
            margin: 2px 2px 2px 0;
            font-family: monospace;
        }
        .lit-type-badge {
            display: inline-block;
            border-radius: 4px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            color: white;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for paper in LITERATURE:
        typeColor = TYPE_COLORS.get(paper["type"], "#888")
        pdfPath = os.path.join("literatureDocs", paper["pdf"])

        with st.container(border=True):
            headerCol, badgeCol = st.columns([10, 2])
            with headerCol:
                st.markdown(
                    f"<span class='lit-type-badge' style='background:{typeColor};'>{paper['type']}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {paper['id']}. {paper['title']}")
                st.markdown(
                    f"**{paper['authors']}** &nbsp;·&nbsp; *{paper['journal']}* &nbsp;·&nbsp; **{paper['year']}**",
                    unsafe_allow_html=True,
                )
                tagHtml = " ".join(
                    f"<span class='lit-tag'>{t}</span>" for t in paper["tags"]
                )
                st.markdown(tagHtml, unsafe_allow_html=True)

            with badgeCol:
                st.link_button("Open Paper", paper["url"], use_container_width=True)
                if os.path.exists(pdfPath):
                    st.download_button(
                        "Download PDF",
                        data=_pdfBytes(pdfPath),
                        file_name=paper["pdf"],
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"dl_{paper['id']}",
                    )

            st.markdown("**Overview**")
            st.write(paper["overview"])

            detailCol1, detailCol2 = st.columns(2)

            with detailCol1:
                with st.expander("Primary Aim"):
                    st.write(paper["aim"])
                with st.expander("Methods & Algorithms"):
                    st.markdown(paper["methods"])

            with detailCol2:
                with st.expander("Key Results"):
                    st.markdown(paper["results"])
                with st.expander("Conclusion"):
                    st.write(paper["conclusion"])

            with st.expander("Full Reference (APA)"):
                st.code(paper["reference"], language=None)
