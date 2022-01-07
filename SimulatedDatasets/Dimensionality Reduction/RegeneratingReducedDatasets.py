import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
import seaborn as sns


new_df1 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_CC_modified.csv", index_col=0)
new_df2 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_Ia_modified.csv", index_col=0)
new_df3 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_SESNe_modified.csv", index_col=0)


def phasecut(df, n):
    df["before" + str(n) + "Days"] = [1 if row[1][0] < n else 0 for row in df.iterrows()]


def Before_after_array(X, y):
    Before_list = []
    After_list = []
    for i in range(len(y)):
        if y[i] == 1.0:
            Before_list.append(X[i])
        else:
            After_list.append(X[i])
    Before_array = np.array(Before_list)
    After_array = np.array(After_list)
    return Before_array, After_array

def tsne_and_umap_plotter(new_df, title):
    method_tsne_1 = TSNE(n_components=2, learning_rate=200, perplexity=25)
    method_tsne_2 = TSNE(n_components=2, learning_rate=500 ,perplexity=30)
    method_umap_1 = umap.UMAP(n_neighbors=15, min_dist=0.5)
    method_umap_2 = umap.UMAP(n_neighbors=50, min_dist=0.1)
    day = 3
    phasecut(new_df, day)


    data = new_df.to_numpy()
    X = data[:, 1:8].astype('float')
    y = data[:, 9].astype('float')
    label = data[:, 0].astype('float')
    tsne_X1_embedded = method_tsne_1.fit_transform(X)
    tsne_X1_embedded = np.append(label.reshape((np.shape(data)[0], 1)), tsne_X1_embedded, axis=1)

    tsne_X2_embedded = method_tsne_2.fit_transform(X)
    tsne_X2_embedded = np.append(label.reshape((np.shape(data)[0], 1)), tsne_X2_embedded, axis=1)

    umap_X1_embedded = method_umap_1.fit_transform(X)
    umap_X1_embedded = np.append(label.reshape((np.shape(data)[0], 1)), umap_X1_embedded, axis=1)

    umap_X2_embedded = method_umap_2.fit_transform(X)
    umap_X2_embedded = np.append(label.reshape((np.shape(data)[0], 1)), umap_X2_embedded, axis=1)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    sns.scatterplot(data=tsne_X1_embedded, x=tsne_X1_embedded[:, 1], y=tsne_X1_embedded[:, 2], hue=tsne_X1_embedded[:, 0], ax=ax[0][0])
    ax[0][0].set_title("T-SNE Perplexity 25 for " + title + " dataset")

    sns.scatterplot(data=tsne_X2_embedded, x=tsne_X2_embedded[:, 1], y=tsne_X2_embedded[:, 2], hue=tsne_X2_embedded[:, 0], ax=ax[1][0])
    ax[1][0].set_title("T-SNE Learning Rate 500 for " + title + " dataset")

    sns.scatterplot(data=umap_X1_embedded, x=umap_X1_embedded[:, 1], y=umap_X1_embedded[:, 2], hue=umap_X1_embedded[:, 0], ax=ax[0][1])
    ax[0][1].set_title("UMAP Minimum Distance 0.5 for " + title + " dataset")

    sns.scatterplot(data=umap_X2_embedded, x=umap_X2_embedded[:, 1], y=umap_X2_embedded[:, 2], hue=umap_X2_embedded[:, 0], ax=ax[1][1])
    ax[1][1].set_title("UMAP Nearest Neighbours 50 for " + title + " dataset")

    plt.suptitle("Dimentionality Reduction on " + title + " dataset")
    plt.savefig("Dimentionality Reduction on " + title + " dataset.jpg")

dataframes = [new_df1, new_df2, new_df3]
titles = ["CC", "Ia", "SESNe"]

for (dataframe, title) in zip(dataframes, titles):
    tsne_and_umap_plotter(dataframe, title)