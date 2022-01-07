import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

def PCALoadingsPlotter(df, title, scaling, i):
    pca = PCA(n_components=2)
    dataset = pca.fit_transform(df.to_numpy()[:, 1:8])

    data = pca.components_.T * np.sqrt(pca.explained_variance_)
    labels = ['g', 'r', 'i', 'z', 'g-r', 'r-i', 'i-z']
    colors = ['green', 'red', 'crimson', 'black', 'magenta', 'gray', 'purple']

    ax[i].plot(dataset[:, 0], dataset[:, 1], '.', mec='navy')

    origin = [0, 0]

    for j in range(7):
        ax[i].quiver(*origin, data[j, 0], data[j, 1], label=labels[j], color=colors[j], scale=scaling, zorder=100)
    ax[i].set_xlim(-10, 30)
    ax[i].set_ylim(-10, 15)
    ax[i].axhline(0, color='black')
    ax[i].axvline(0, color='black')
    ax[i].legend(loc='best')
    ax[i].set_title(title)

def PCALoadingswithParametersPlot(df):
    pca = PCA(n_components=2)
    dataset = pca.fit_transform(df.to_numpy()[:, 1:8])

    data = pca.components_.T * np.sqrt(pca.explained_variance_)
    labels = ['g', 'r', 'i', 'z', 'g-r', 'r-i', 'i-z']
    colors = ['green', 'red', 'crimson', 'black', 'magenta', 'gray', 'purple']

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    sns.scatterplot(data=dataset, x=dataset[:, 0], y=dataset[:, 1], hue=df.to_numpy()[:, 8], ax=axes[0])
    axes[0].set_title("Supernovae types")

    sns.scatterplot(data=dataset, x=dataset[:, 0], y=dataset[:, 1], hue=df.to_numpy()[:, 0], ax=axes[1])
    axes[1].set_title("Days after explosion")

    origin = [0, 0]
    vectors = []

    for i in range(2):
        for j in range(7):
            vec = axes[i].quiver(*origin, data[j, 0], data[j, 1], color=colors[j], scale=4, zorder=100)
            vectors.append(vec)
        legend = axes[i].legend(vectors, labels, loc='upper left')
        axes[i].add_artist(legend)
        axes[i].set_xlim(-10, 35)
        axes[i].set_ylim(-15, 20)
        axes[i].axhline(0, color='black')
        axes[i].axvline(0, color='black')
        axes[i].legend(loc='best')

    plt.suptitle("PCA Loadings with Parameters")
    plt.savefig("PCA Loadings with Parameters.jpg")
    plt.show()


new_df1 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_CC_modified.csv", index_col=0)
new_df2 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_Ia_modified.csv", index_col=0)
new_df3 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_SESNe_modified.csv", index_col=0)

dataframes = [new_df1, new_df2, new_df3]
titles = ["Core Collapse", "Type Ia", "Striped Envelope"]
scalings = [3, 5, 4.5]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

for i in range(len(dataframes)):
    PCALoadingsPlotter(dataframes[i], titles[i], scalings[i], i)

plt.tight_layout()
plt.savefig("PCA Loadings for each dataset")
plt.show()

final_df = pd.concat([new_df1, new_df2, new_df3])
PCALoadingswithParametersPlot(final_df)