dataPath = "ev_battery_degradation_v1.csv"
figuresDir = "outputs/figures"
modelsDir = "outputs/models"

randomState = 42
testSize = 0.2

target = "SoH_Percent"

dropCols = ["Vehicle_ID", "Battery_Status"]

numericFeatures = [
    "Battery_Capacity_kWh",
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
    "Internal_Resistance_Ohm",
]

ordinalCol = "Driving_Style"
ordinalMap = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}

onehotCols = ["Car_Model", "Battery_Type"]

electroDrivers = ["Internal_Resistance_Ohm", "Avg_Temperature_C"]
