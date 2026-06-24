import config
import preprocessing
import eda
import models
import evaluate


data = preprocessing.loadData()

print("Making EDA charts...")
eda.generateAllEda(data, config.figuresDir)

print("Preprocessing...")
xTrain, xTest, yTrain, yTest, scaler, featureCols = preprocessing.preprocess(data)

print("Training models...")
allModels, bestName = models.trainAndPickBest(xTrain, yTrain)

print("Evaluating...")
results = {}
for name in allModels:
    results[name] = evaluate.evaluateModel(
        allModels[name], xTest, yTest, featureCols,
        figuresDir=config.figuresDir, modelName=name,
    )

models.saveArtifacts(allModels[bestName], scaler, featureCols, config.modelsDir)

print("\nModel comparison (sorted by RMSE):")
print(evaluate.summarizeMetrics(results))
print("\nBest model:", bestName)
print("Figures saved to:", config.figuresDir)
print("Models saved to:", config.modelsDir)
