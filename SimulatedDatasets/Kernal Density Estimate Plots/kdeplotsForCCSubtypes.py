import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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

def Generate_Color_Subplots(ax, array, i):
    sns.kdeplot(data=array, x=array[:, 1], hue=array[:, 0], ax=ax[i, 0])
    ax[i, 0].set_xlim(12.5, 25)
    sns.kdeplot(data=array, x=array[:, 2], hue=array[:, 0], ax=ax[i, 1])
    ax[i, 1].set_xlim(12.5, 25)
    sns.kdeplot(data=array, x=array[:, 3], hue=array[:, 0], ax=ax[i, 2])
    ax[i, 2].set_xlim(12.5, 25)
    sns.kdeplot(data=array, x=array[:, 4], hue=array[:, 0], ax=ax[i, 3])
    ax[i, 3].set_xlim(12.5, 25)

new_df1 = pd.read_csv("SimulatedDatasets/LCs_CC_modified.csv", index_col=0)
SNII_list = []
IIb_list = []
IIL_list = []
IIn_list = []


for row in new_df1.iterrows():
    if row[1][8] == "SNII":
        SNII_list.append(row[1].to_numpy())
    elif row[1][8] == "IIL":
        IIL_list.append(row[1].to_numpy())
    elif row[1][8] == "IIb":
        IIb_list.append(row[1].to_numpy())
    elif row[1][8] == "IIn":
        IIn_list.append(row[1].to_numpy())

SNII_array = np.array(SNII_list)
IIn_array = np.array(IIn_list)
IIb_array = np.array(IIb_list)
IIL_array = np.array(IIL_list)

SNII_df = pd.DataFrame(SNII_array, columns=['Days after explosion', 'g mag', 'r mag', 'i mag', 'z mag', 'g-r color', 'r-i color', 'i-z color', 'Type'])
print(SNII_df.head())

fig1, ax1 = plt.subplots(nrows=4, ncols=4, figsize=(12, 8))
Generate_Color_Subplots(ax1, SNII_array, 0)
Generate_Color_Subplots(ax1, IIb_array, 1)
Generate_Color_Subplots(ax1, IIL_array, 2)
Generate_Color_Subplots(ax1, IIn_array, 3)
plt.suptitle("KDEPlots for different magnitudes for subtypes in CC dataset")
plt.savefig("KDEPlots for different magnitudes for subtypes in CC dataset.jpg")
plt.tight_layout()
plt.show()

Generate_Color_and_Mag_KDEplots(SNII_df, "SNII")

