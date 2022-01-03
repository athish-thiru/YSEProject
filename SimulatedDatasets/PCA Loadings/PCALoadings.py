import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def PCALoadingsPlotter(df, title, i):
    pca = PCA(n_components=2)
    dataset = pca.fit_transform(df.to_numpy()[:, 1:8])

    data = pca.components_.T * np.sqrt(pca.explained_variance_)
    labels = ['g', 'r', 'i', 'z', 'g-r', 'r-i', 'i-z']
    colors = ['green', 'red', 'crimson', 'black', 'magenta', 'gray', 'purple']

    ax[i].plot(dataset[:, 0], dataset[:, 1], '.', mec='navy')

    origin = [0, 0]

    for j in range(7):
        ax[i].quiver(*origin, data[j, 0], data[j, 1], label=labels[j], color=colors[j], scale=4, zorder=100)
    ax[i].set_xlim(-10, 30)
    ax[i].set_ylim(-10, 15)
    ax[i].axhline(0, color='black')
    ax[i].axvline(0, color='black')
    ax[i].legend(loc='best')
    ax[i].set_title(title)

new_df1 = pd.read_csv("SimulatedDatasets/LCs_CC_modified.csv", index_col=0)
new_df2 = pd.read_csv("SimulatedDatasets/LCs_Ia_modified.csv", index_col=0)
new_df3 = pd.read_csv("SimulatedDatasets/LCs_SESNe_modified.csv", index_col=0)

dataframes = [new_df1, new_df2, new_df3]
titles = ["Core Collapse", "Type Ia", "Striped Envelope"]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

for i in range( len(dataframes)):
    PCALoadingsPlotter(dataframes[i], titles[i], i)

plt.tight_layout()
plt.savefig("PCA Loadings for each dataset")
plt.show()
