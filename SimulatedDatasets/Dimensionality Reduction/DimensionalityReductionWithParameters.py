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


def tsne_embedder(data, perplexity, rate):
    method = TSNE(n_components=2, learning_rate=rate, perplexity=perplexity)
    X = data[:, 1:8].astype('float')
    label = data[:, 0].astype('float')
    X_embedded = method.fit_transform(X)
    X_embedded = np.append(label.reshape((np.shape(data)[0], 1)), X_embedded, axis=1)
    return X_embedded


def umap_embedder(data,n_neighbours, min_dist):
    method = umap.UMAP(n_neighbors=n_neighbours, min_dist=min_dist)
    X = data[:, 1:8].astype('float')
    label = data[:, 0].astype('float')
    X_embedded = method.fit_transform(X)
    X_embedded = np.append(label.reshape((np.shape(data)[0], 1)), X_embedded, axis=1)
    return X_embedded


def umap_and_tsne_plotter(new_df, title):
    day = 3
    phasecut(new_df, day)

    data = new_df.to_numpy()
    X1 = data[:, 1:8].astype('float')
    y1 = data[:, 9].astype('float')
    label = data[:, 0].astype('float')

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))
    learning_rates = [10, 200, 400, 600, 800, 1000]
    perplexities = [5, 10, 20, 30, 40, 50]
    n_neighbours = [2, 20, 40, 60, 80, 100]
    min_dists = [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]

    for i in range(2):
        for j in range(3):
            X_embedded = tsne_embedder(data, 30, learning_rates[(3*i) + j])
            ax[i][j].set_xlim(-20, 20)
            ax[i][j].set_ylim(-20, 20)
            sns.scatterplot(data=X_embedded, x=X_embedded[:, 1], y=X_embedded[:, 2], hue=X_embedded[:, 0], ax=ax[i][j])
            ax[i][j].set_title("Learning Rate: {}".format(learning_rates[(3*i) + j]))

    plt.suptitle("TSNE on " + title + " dataset with Learning Rate variation")
    plt.savefig("TSNE on " + title + " dataset with Learning Rate variation.jpg")

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))
    for i in range(2):
        for j in range(3):
            X_embedded = tsne_embedder(data, perplexities[(3*i) + j], 200)
            ax[i][j].set_xlim(-80, 80)
            ax[i][j].set_ylim(-80, 80)
            ax[i][j] = sns.scatterplot(data=X_embedded, x=X_embedded[:, 1], y=X_embedded[:, 2], hue=X_embedded[:, 0], ax=ax[i][j])
            ax[i][j].set_title("Perplexity: {}".format(perplexities[(3*i) + j]))
    plt.suptitle("TSNE on " + title + " dataset with Perplexity variation")
    plt.savefig("TSNE on " + title + " dataset with Perplexity variation.jpg")

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))
    for i in range(2):
        for j in range(3):
            X_embedded = umap_embedder(data, n_neighbours[(3*i) + j], 0.1)
            ax[i][j].set_xlim(-12, 15)
            ax[i][j].set_ylim(-12, 12)
            sns.scatterplot(data=X_embedded, x=X_embedded[:, 1], y=X_embedded[:, 2], hue=X_embedded[:, 0], ax=ax[i][j])
            ax[i][j].set_title("Nearest Neighbours: {}".format(n_neighbours[(3*i) + j]))
    plt.suptitle("UMAP on " + title + " dataset with Nearest Neighbours variation")
    plt.savefig("UMAP on " + title + " dataset with Nearest Neighbours variation.jpg")

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 10))
    for i in range(2):
        for j in range(3):
            X_embedded = umap_embedder(data, 12, min_dists[(3*i) + j])
            ax[i][j].set_xlim(-15, 15)
            ax[i][j].set_ylim(-15, 15)
            sns.scatterplot(data=X_embedded, x=X_embedded[:, 1], y=X_embedded[:, 2], hue=X_embedded[:, 0], ax=ax[i][j])
            ax[i][j].set_title("Minimum Distance: {}".format(min_dists[(3*i) + j]))
    plt.suptitle("UMAP on " + title + " dataset with Minimum Distance variation")
    plt.savefig("UMAP on " + title + " dataset with Minimum Distance variation.jpg")


dataframes = [new_df2, new_df3]
titles = ["Ia", "SESNe"]

# for (dataframe, title) in zip(dataframes, titles):
#     umap_and_tsne_plotter(dataframe, title)

umap_and_tsne_plotter(new_df3, "SESNe")