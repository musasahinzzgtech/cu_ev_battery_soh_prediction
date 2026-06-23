# Loading the data and turning it into numbers for the models.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def loadData(path=config.dataPath):
    # read the csv file
    df = pd.read_csv(path)

    # quick check that the columns we need are there
    needed = config.dropCols + config.numericFeatures + [config.ordinalCol]
    needed = needed + config.onehotCols + [config.target]
    for col in needed:
        if col not in df.columns:
            print("WARNING: missing column", col)

    print("Loaded", df.shape[0], "rows and", df.shape[1], "columns")
    return df


def makeFeatures(df):
    # turn the raw dataframe into the X feature table (encoded, not scaled)
    df = df.copy()

    # drop id/target columns if they are there
    for col in config.dropCols + [config.target]:
        if col in df.columns:
            df = df.drop(columns=col)

    # driving style -> 0/1/2
    df[config.ordinalCol] = df[config.ordinalCol].map(config.ordinalMap)

    # one-hot encode the text columns
    df = pd.get_dummies(df, columns=config.onehotCols, drop_first=True, dtype=float)
    return df


def preprocess(df):
    # full preprocessing: features -> split -> scale numbers
    y = df[config.target].astype(float)
    X = makeFeatures(df)

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y, test_size=config.testSize, random_state=config.randomState
    )

    # scale the numeric columns (fit only on the training data)
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
