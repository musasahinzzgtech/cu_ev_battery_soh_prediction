# Makes the charts we use to look at the data (saved as png files).

import os

import matplotlib
matplotlib.use("Agg")  # so it works without a screen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid", context="paper")


def saveFig(fig, savePath, title):
    # add a title, save the figure and close it
    folder = os.path.dirname(savePath)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savePath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", savePath)


def plotFeatureDistributions(df, savePath):
    # histograms of the target and 3 important columns
    panels = [
        (config.target, "State of Health (%)"),
        ("Total_Charging_Cycles", "Total Charging Cycles"),
        ("Avg_Temperature_C", "Average Temperature (°C)"),
        ("Internal_Resistance_Ohm", "Internal Resistance (Ohm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (column, label) in zip(axes.ravel(), panels):
        sns.histplot(data=df, x=column, kde=True, color="#2c3e50", ax=ax)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
    saveFig(fig, savePath, "Feature Distributions (Univariate Analysis)")


def plotSohBoxplotByChemistry(df, savePath):
    # box plot of SoH for LFP vs NMC
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x="Battery_Type",
        y=config.target,
        hue="Battery_Type",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Battery Type")
    ax.set_ylabel("State of Health (%)")
    saveFig(fig, savePath, "SoH Distribution by Battery Chemistry")


def plotTemperatureVsSoh(df, savePath):
    # scatter of temperature vs SoH, colored by battery type
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Avg_Temperature_C",
        y=config.target,
        hue="Battery_Type",
        alpha=0.4,
        s=18,
        edgecolor=None,
        ax=ax,
    )
    for chemistry, group in df.groupby("Battery_Type"):
        sns.regplot(
            data=group,
            x="Avg_Temperature_C",
            y=config.target,
            scatter=False,
            ax=ax,
            line_kws={"linewidth": 2},
            label=str(chemistry) + " trend",
        )
    ax.set_xlabel("Average Temperature (°C)")
    ax.set_ylabel("State of Health (%)")
    ax.legend(title="Battery Type", loc="best")
    saveFig(fig, savePath, "Thermal Effect: Average Temperature vs. SoH by Chemistry")


def plotCorrelationHeatmap(df, savePath):
    # heatmap of correlations between numeric columns and the target
    numericCols = config.numericFeatures + [config.target]
    corr = df[numericCols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    saveFig(fig, savePath, "Feature Correlation Heatmap")


def plotCyclesVsSoh(df, savePath):
    # scatter of charging cycles vs SoH, colored by battery type
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Total_Charging_Cycles",
        y=config.target,
        hue="Battery_Type",
        alpha=0.4,
        s=18,
        edgecolor=None,
        ax=ax,
    )
    for chemistry, group in df.groupby("Battery_Type"):
        sns.regplot(
            data=group,
            x="Total_Charging_Cycles",
            y=config.target,
            scatter=False,
            order=2,
            ax=ax,
            line_kws={"linewidth": 2},
            label=str(chemistry) + " trend",
        )
    ax.set_xlabel("Total Charging Cycles")
    ax.set_ylabel("State of Health (%)")
    ax.legend(title="Battery Type", loc="best")
    saveFig(fig, savePath, "Capacity Fade: Charging Cycles vs. SoH by Chemistry")


def plotArrheniusDegradation(df, savePath):
    # Arrhenius plot: shows that higher temperature speeds up degradation.
    # rate = SoH loss per month; ln(rate) vs 1000/T should be roughly a line.
    gasConstant = 8.314  # J / (mol K)
    kelvinOffset = 273.15

    work = df.copy()
    work = work[work["Vehicle_Age_Months"] > 0].copy()
    work["degRate"] = (100.0 - work[config.target]) / work["Vehicle_Age_Months"]
    work = work[work["degRate"] > 0].copy()
    work["tempKelvin"] = work["Avg_Temperature_C"] + kelvinOffset
    work["invTemp"] = 1.0 / work["tempKelvin"]

    # group temperatures into bins to reduce noise
    work["tempBin"] = pd.cut(work["Avg_Temperature_C"], bins=12)
    grouped = (
        work.groupby("tempBin", observed=True)
        .agg(meanRate=("degRate", "mean"), meanInvTemp=("invTemp", "mean"))
        .dropna()
    )
    grouped = grouped[grouped["meanRate"] > 0]

    x = (1000.0 * grouped["meanInvTemp"]).to_numpy()
    y = np.log(grouped["meanRate"].to_numpy())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=45, color="#c0392b", label="Binned data")

    annotation = "Not enough spread to fit a line"
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xLine = np.linspace(x.min(), x.max(), 100)
        ax.plot(xLine, slope * xLine + intercept, color="#2c3e50",
                linewidth=2, label="Arrhenius fit")
        # slope = -Ea / (1000 * R), so Ea (kJ/mol):
        activationEnergyKj = -slope * gasConstant
        annotation = ("Apparent Ea = " + str(round(activationEnergyKj, 1)) +
                      " kJ/mol\nslope = " + str(round(slope, 3)))

    ax.set_xlabel("1000 / T  (1/K)")
    ax.set_ylabel("ln(degradation rate)")
    ax.text(0.05, 0.05, annotation, transform=ax.transAxes, fontsize=11,
            va="bottom", bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    ax.legend(loc="upper right")
    saveFig(fig, savePath, "Arrhenius Plot: Thermal Acceleration of Degradation")


def generateAllEda(df, figuresDir):
    # make all the charts and save them
    os.makedirs(figuresDir, exist_ok=True)
    plotFeatureDistributions(df, os.path.join(figuresDir, "feature_distributions.png"))
    plotSohBoxplotByChemistry(df, os.path.join(figuresDir, "soh_boxplot_by_chemistry.png"))
    plotTemperatureVsSoh(df, os.path.join(figuresDir, "temperature_vs_soh.png"))
    plotCorrelationHeatmap(df, os.path.join(figuresDir, "correlation_heatmap.png"))
    plotCyclesVsSoh(df, os.path.join(figuresDir, "cycles_vs_soh.png"))
    plotArrheniusDegradation(df, os.path.join(figuresDir, "arrhenius_degradation.png"))


if __name__ == "__main__":
    data = pd.read_csv(config.dataPath)
    generateAllEda(data, config.figuresDir)
