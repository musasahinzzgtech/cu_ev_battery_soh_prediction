# Runs the whole project: load data, make charts, train models, evaluate, save.

import config
import preprocessing
import eda
import models
import evaluate


# 1. load the data
data = preprocessing.loadData()

# 2. make the EDA charts
print("Making EDA charts...")
eda.generateAllEda(data, config.figuresDir)

# 3. preprocess (encode + scale + split)
print("Preprocessing...")
xTrain, xTest, yTrain, yTest, scaler, featureCols = preprocessing.preprocess(data)

# 4. train the models and pick the best
print("Training models...")
allModels, bestName = models.trainAndPickBest(xTrain, yTrain)

# 5. evaluate every model on the test set
print("Evaluating...")
results = {}
for name in allModels:
    results[name] = evaluate.evaluateModel(
        allModels[name], xTest, yTest, featureCols,
        figuresDir=config.figuresDir, modelName=name,
    )

# 6. save the best model (and the scaler + feature list)
models.saveArtifacts(allModels[bestName], scaler, featureCols, config.modelsDir)

# show a summary table
print("\nModel comparison (sorted by RMSE):")
print(evaluate.summarizeMetrics(results))
print("\nBest model:", bestName)
print("Figures saved to:", config.figuresDir)
print("Models saved to:", config.modelsDir)
