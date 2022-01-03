import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
import seaborn as sns


new_df1 = pd.read_csv("SimulatedDatasets/LCs_CC_modified.csv", index_col=0)
new_df2 = pd.read_csv("SimulatedDatasets/LCs_Ia_modified.csv", index_col=0)
new_df3 = pd.read_csv("SimulatedDatasets/LCs_SESNe_modified.csv", index_col=0)


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


# t-SNE dimensionality reduction
method_tsne_1 = TSNE(n_components=2, learning_rate=800)
method_tsne_2 = TSNE(n_components=2, perplexity=100)
method_tsne_3 = TSNE(n_components=2, perplexity=50, learning_rate=800)
method_umap_1 = umap.UMAP(n_neighbors=80)
method_umap_2 = umap.UMAP(min_dist=0.0)
method_umap_3 = umap.UMAP(n_components=1)
day = 3
phasecut(new_df1, day)
phasecut(new_df2, day)
phasecut(new_df3, day)

data1 = new_df1.to_numpy()
X1 = data1[:, 1:8].astype('float')
y1 = data1[:, 9].astype('float')
label = data1[:, 0].astype('float')
tsne_X1_embedded = method_tsne_1.fit_transform(X1)
tsne_X1_embedded = np.append(label.reshape((np.shape(data1)[0], 1)), tsne_X1_embedded, axis=1)

tsne_X2_embedded = method_tsne_2.fit_transform(X1)
tsne_X2_embedded = np.append(label.reshape((np.shape(data1)[0], 1)), tsne_X2_embedded, axis=1)

tsne_X3_embedded = method_tsne_3.fit_transform(X1)
tsne_X3_embedded = np.append(label.reshape((np.shape(data1)[0], 1)), tsne_X3_embedded, axis=1)

umap_X1_embedded = method_umap_1.fit_transform(X1)
umap_X1_embedded = np.append(label.reshape((np.shape(data1)[0], 1)), umap_X1_embedded, axis=1)

umap_X2_embedded = method_umap_2.fit_transform(X1)
umap_X2_embedded = np.append(label.reshape((np.shape(data1)[0], 1)), umap_X2_embedded, axis=1)

umap_X3_embedded = method_umap_3.fit_transform(X1)
umap_X3_embedded = np.append(label.reshape((np.shape(data1)[0], 1)), umap_X3_embedded, axis=1)

sns.scatterplot(data=tsne_X1_embedded, x=tsne_X1_embedded[:, 1], y=tsne_X1_embedded[:, 2], hue=tsne_X1_embedded[:, 0])
plt.title("T-SNE increased learning rate")
plt.show()

sns.scatterplot(data=tsne_X2_embedded, x=tsne_X2_embedded[:, 1], y=tsne_X2_embedded[:, 2], hue=tsne_X2_embedded[:, 0])
plt.title("T-SNE higher perplexity")
plt.show()

sns.scatterplot(data=tsne_X3_embedded, x=tsne_X3_embedded[:, 1], y=tsne_X3_embedded[:, 2], hue=tsne_X3_embedded[:, 0])
plt.title("T-SNE higher perplexity and higher learning rate")
plt.show()

sns.scatterplot(data=umap_X1_embedded, x=umap_X1_embedded[:, 1], y=umap_X1_embedded[:, 2], hue=umap_X1_embedded[:, 0])
plt.title("UMAP nearest neighbours")
plt.show()

sns.scatterplot(data=umap_X2_embedded, x=umap_X2_embedded[:, 1], y=umap_X2_embedded[:, 2], hue=umap_X2_embedded[:, 0])
plt.title("UMAP minimum distance")
plt.show()

sns.scatterplot(data=umap_X3_embedded, x=umap_X3_embedded[:, 1], y=umap_X3_embedded[:, 2], hue=umap_X3_embedded[:, 0])
plt.title("UMAP 1 component")
plt.show()


# data2 = new_df2.to_numpy()
# X2 = data2[:, 1:8].astype('float')
# y2 = data2[:, 9].astype('float')
# X2_embedded = method.fit_transform(X2)
#
# data3 = new_df3.to_numpy()
# X3 = data3[:, 1:8].astype('float')
# y3 = data3[:, 9].astype('float')
# X3_embedded = method.fit_transform(X3)
#
# #UMAP dimensionality reduction
#

#

#
# umap_X2 = data2[:, 1:8].astype('float')
# umap_y2 = data2[:,9].astype('float')
# umap_X2_embedded = method2.fit_transform(X2)
#
# umap_X3 = data3[:, 1:8].astype('float')
# umap_y3 = data3[:,9].astype('float')
# umap_X3_embedded = method2.fit_transform(X3)
#
# print(X1_embedded)
# print(X2_embedded)
# print(X3_embedded)
#
# print(umap_X1_embedded)
# print(umap_X2_embedded)
# print(umap_X3_embedded)