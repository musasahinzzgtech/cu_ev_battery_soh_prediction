import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def loadData(path=config.dataPath):
    df = pd.read_csv(path)

    needed = config.dropCols + config.numericFeatures + [config.ordinalCol]
    needed = needed + config.onehotCols + [config.target]
    for col in needed:
        if col not in df.columns:
            print("WARNING: missing column", col)

    print("Loaded", df.shape[0], "rows and", df.shape[1], "columns")
    return df


def makeFeatures(df):
    df = df.copy()

    for col in config.dropCols + [config.target]:
        if col in df.columns:
            df = df.drop(columns=col)

    df[config.ordinalCol] = df[config.ordinalCol].map(config.ordinalMap)

    df = pd.get_dummies(df, columns=config.onehotCols, drop_first=True, dtype=float)
    return df


def preprocess(df):
    y = df[config.target].astype(float)
    X = makeFeatures(df)

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=config.testSize, random_state=config.randomState
    )

    scaler = StandardScaler()
    xTrain = xTrain.copy()
    xTest = xTest.copy()
    xTrain[config.numericFeatures] = scaler.fit_transform(
        xTrain[config.numericFeatures]
    )
    xTest[config.numericFeatures] = scaler.transform(xTest[config.numericFeatures])

    featureCols = list(xTrain.columns)
    print("Train:", xTrain.shape[0], "Test:", xTest.shape[0],
          "Features:", len(featureCols))
    return xTrain, xTest, yTrain, yTest, scaler, featureCols


if __name__ == "__main__":
    data = loadData()
    xTr, xTe, yTr, yTe, sc, cols = preprocess(data)
    print("Feature columns:", cols)
