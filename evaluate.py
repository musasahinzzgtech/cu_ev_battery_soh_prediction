# Checks how good a model is (numbers + charts).

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


def computeMetrics(yTrue, yPred):
    # MAE, RMSE and R2
    yTrue = np.asarray(yTrue, dtype=float)
    yPred = np.asarray(yPred, dtype=float)
    mae = float(mean_absolute_error(yTrue, yPred))
    rmse = float(np.sqrt(mean_squared_error(yTrue, yPred)))
    r2 = float(r2_score(yTrue, yPred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def getImportances(model, featureNames):
    # get feature importance from trees, or coefficients from linear models
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    s = pd.Series(values, index=list(featureNames))
    return s.sort_values(ascending=False)


def plotPredictedVsActual(yTrue, yPred, savePath, modelName="Model"):
    # scatter of predicted vs actual SoH
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    yTrue = np.asarray(yTrue, dtype=float)
    yPred = np.asarray(yPred, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(yTrue, yPred, alpha=0.4, s=18, edgecolor=None)
    low = min(yTrue.min(), yPred.min())
    high = max(yTrue.max(), yPred.max())
    ax.plot([low, high], [low, high], color="#c0392b", linewidth=2, label="Ideal (y = x)")
    ax.set_xlabel("Actual SoH (%)")
    ax.set_ylabel("Predicted SoH (%)")
    ax.set_title(modelName + ": Predicted vs. Actual SoH", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(savePath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", savePath)


def plotResiduals(yTrue, yPred, savePath, modelName="Model"):
    # residuals (actual - predicted) vs predicted
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    yTrue = np.asarray(yTrue, dtype=float)
    yPred = np.asarray(yPred, dtype=float)
    residuals = yTrue - yPred

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(yPred, residuals, alpha=0.4, s=18, edgecolor=None)
    ax.axhline(0.0, color="#c0392b", linewidth=2)
    ax.set_xlabel("Predicted SoH (%)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(modelName + ": Residual Plot", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savePath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", savePath)


def plotFeatureImportance(model, featureNames, savePath, modelName="Model", topN=15):
    # bar chart of the most important features (chemistry ones in red)
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    importances = getImportances(model, featureNames)
    if importances is None:
        print("No feature importance for", modelName)
        return

    # print where the chemistry features ended up
    ranked = list(importances.index)
    for driver in config.electroDrivers:
        if driver in ranked:
            print("Driver", driver, "ranked #" + str(ranked.index(driver) + 1))

    top = importances.head(topN).iloc[::-1]
    colors = ["#c0392b" if f in config.electroDrivers else "#34495e" for f in top.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("Importance")
    ax.set_title(modelName + ": Feature Importances\n(chemistry drivers in red)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savePath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", savePath)


def evaluateModel(model, xTest, yTest, featureNames, figuresDir=config.figuresDir,
                  modelName="Model"):
    # predict on the test set, print metrics, and save the charts
    yPred = model.predict(xTest)
    metrics = computeMetrics(yTest, yPred)
    print(modelName, "-> MAE:", round(metrics["MAE"], 4),
          "RMSE:", round(metrics["RMSE"], 4), "R2:", round(metrics["R2"], 4))

    slug = modelName.lower().replace(" ", "_")
    plotPredictedVsActual(yTest, yPred,
                          os.path.join(figuresDir, slug + "_pred_vs_actual.png"), modelName)
    plotResiduals(yTest, yPred,
                  os.path.join(figuresDir, slug + "_residuals.png"), modelName)
    plotFeatureImportance(model, featureNames,
                          os.path.join(figuresDir, slug + "_feature_importance.png"), modelName)
    return metrics


def summarizeMetrics(allMetrics):
    # make a small table of all the models, sorted by RMSE
    table = pd.DataFrame(allMetrics).T
    return table.sort_values("RMSE")
