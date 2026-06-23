# Settings and column names used by the whole project.

# file paths
dataPath = "ev_battery_degradation_v1.csv"
figuresDir = "outputs/figures"
modelsDir = "outputs/models"

# train/test split
randomState = 42
testSize = 0.2

# what we want to predict
target = "SoH_Percent"

# columns we don't use as features (id + a column that leaks the answer)
dropCols = ["Vehicle_ID", "Battery_Status"]

# numeric columns
numericFeatures = [
    "Battery_Capacity_kWh",
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
    "Internal_Resistance_Ohm",
]

# driving style has an order, so we turn it into numbers
ordinalCol = "Driving_Style"
ordinalMap = {"Conservative": 0, "Moderate": 1, "Aggressive": 2}

# text columns we one-hot encode
onehotCols = ["Car_Model", "Battery_Type"]

# the two features that come from battery chemistry
electroDrivers = ["Internal_Resistance_Ohm", "Avg_Temperature_C"]
