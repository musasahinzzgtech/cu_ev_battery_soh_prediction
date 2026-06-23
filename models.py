# Trains the models and saves the best one.

import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

import config

# xgboost needs an extra system library, so import it only if it works
try:
    from xgboost import XGBRegressor
    hasXgboost = True
except Exception:
    hasXgboost = False

cvFolds = 5
scoring = "neg_root_mean_squared_error"


def trainLinear(xTrain, yTrain):
    # simple linear regression (our baseline)
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    print("Trained Linear Regression")
    return model


def trainRandomForest(xTrain, yTrain):
    # random forest, with a small grid search to find good settings
    paramGrid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=config.randomState, n_jobs=-1),
        paramGrid,
        scoring=scoring,
        cv=cvFolds,
        n_jobs=-1,
    )
    search.fit(xTrain, yTrain)
    print("Random Forest best params:", search.best_params_)
    return search.best_estimator_


def trainXgboost(xTrain, yTrain):
    # gradient boosting model
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.randomState,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(xTrain, yTrain)
    print("Trained XGBoost")
    return model


def cvRmse(model, X, y):
    # average cross-validation RMSE for a model
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cvFolds, n_jobs=-1)
    return float(-np.mean(scores))


def trainAndPickBest(xTrain, yTrain):
    # train all the models and pick the one with the lowest RMSE
    allModels = {}
    allModels["LinearRegression"] = trainLinear(xTrain, yTrain)
    allModels["RandomForest"] = trainRandomForest(xTrain, yTrain)
    if hasXgboost:
        allModels["XGBoost"] = trainXgboost(xTrain, yTrain)
    else:
        print("Skipping XGBoost (library not available)")

    scores = {}
    for name in allModels:
        scores[name] = cvRmse(allModels[name], xTrain, yTrain)
        print(name, "CV RMSE:", round(scores[name], 4))

    bestName = min(scores, key=scores.get)
    print("Best model:", bestName)
    return allModels, bestName


def saveArtifacts(model, scaler, featureCols, modelsDir=config.modelsDir):
    # save the model and the things we need to prepare new data
    os.makedirs(modelsDir, exist_ok=True)
    joblib.dump(model, os.path.join(modelsDir, "model.joblib"))
    prep = {"scaler": scaler, "features": featureCols}
    joblib.dump(prep, os.path.join(modelsDir, "prep.joblib"))
    print("Saved model and prep to", modelsDir)


if __name__ == "__main__":
    import preprocessing

    data = preprocessing.loadData()
    xTr, xTe, yTr, yTe, sc, cols = preprocessing.preprocess(data)
    trainedModels, best = trainAndPickBest(xTr, yTr)
    saveArtifacts(trainedModels[best], sc, cols)
