import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def Generate_Color_and_Mag_KDEplots(df, dataset):
    fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    sns.kdeplot(data=df, x=df['g mag'], hue=df['Days after explosion'], ax=ax1[0, 0])
    ax1[0, 0].set_xlim(12.5, 25)
    sns.kdeplot(data=df, x=df['r mag'], hue=df['Days after explosion'], ax=ax1[0, 1])
    ax1[0, 1].set_xlim(12.5, 25)
    sns.kdeplot(data=df, x=df['i mag'], hue=df['Days after explosion'], ax=ax1[1, 0])
    ax1[1, 0].set_xlim(12.5, 25)
    sns.kdeplot(data=df, x=df['z mag'], hue=df['Days after explosion'], ax=ax1[1, 1])
    ax1[1, 1].set_xlim(12.5, 25)
    plt.suptitle("KDEPlots for different magnitudes in " + dataset)
    plt.savefig("KDEPlots for different magnitudes in " + dataset + ".jpg")
    plt.show()

    fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    sns.kdeplot(data=df, x=df['g-r color'], hue=df['Days after explosion'], ax=ax2[0])
    ax2[0].set_xlim(-1, 1)
    sns.kdeplot(data=df, x=df['r-i color'], hue=df['Days after explosion'], ax=ax2[1])
    ax2[1].set_xlim(-1, 1)
    sns.kdeplot(data=df, x=df['i-z color'], hue=df['Days after explosion'], ax=ax2[2])
    ax2[2].set_xlim(-1, 1)
    plt.suptitle("KDEPlots for different colors in " + dataset)
    plt.savefig("KDEPlots for different colors in " + dataset + ".jpg")
    plt.show()


new_df1 = pd.read_csv("SimulatedDatasets/LCs_CC_modified.csv", index_col=0)
new_df2 = pd.read_csv("SimulatedDatasets/LCs_Ia_modified.csv", index_col=0)
new_df3 = pd.read_csv("SimulatedDatasets/LCs_SESNe_modified.csv", index_col=0)

dataframes = [new_df1, new_df2, new_df3]
datasets = ["CC Dataset", "Ia Dataset", "SESNe Dataset"]

for (df, dataset) in zip(dataframes, datasets):
    Generate_Color_and_Mag_KDEplots(df, dataset)