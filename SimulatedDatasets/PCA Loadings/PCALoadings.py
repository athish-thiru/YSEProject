import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from imblearn.over_sampling import SVMSMOTE

def Phasecut(df, n):
    df["before" + str(n) + "Days"] = [1 if row[1][0] < n else 0 for row in df.iterrows()]

def Resampler(X, y):
    svm = SVMSMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = svm.fit_resample(X, y.astype('int'))
    return X_resampled, y_resampled

def PCAFitAndLoadings(X, y):
    pca = PCA(n_components=2)
    dataset = pca.fit_transform(X)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    return dataset, y, loadings

def PCAFitAndLoadingwithResampling(X, y):
    pca = PCA(n_components=2)

    X_resampled, y_resampled = Resampler(X, y)
    dataset = pca.fit_transform(X_resampled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    return dataset, y_resampled, loadings


def PCALoadingsPlotter(df, title, scaling, i):
    titles = ["Core Collapse", "Type Ia", "Striped Envelope"]
    scalings1 = [3, 5, 4.5]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
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

    plt.tight_layout()
    plt.savefig("PCA Loadings for each dataset")
    plt.show()

def PCALoadingswithParametersPlot(df):
    
    dataset, hue, loadings = PCAFitAndLoadings(df.to_numpy()[:, 1:8], df.to_numpy()[:, 8])

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    PCALoadingsBeforeAfterEachPlot(dataset, df.to_numpy()[:, 8], loadings, axes[0], "Supernovae Tyes")
    PCALoadingsBeforeAfterEachPlot(dataset, df.to_numpy()[:, 0], loadings, axes[1], "Days after explosion")

    plt.suptitle("PCA Loadings with Parameters")
    plt.savefig("PCA Loadings with Parameters")
    plt.show()

def PCALoadingsBeforeAfterEachPlot(dataframe, dataset, hue, loadings, axes, title):
    labels = ['g', 'r', 'i', 'z', 'g-r', 'r-i', 'i-z']
    colors = ['green', 'red', 'crimson', 'black', 'magenta', 'gray', 'purple']

    #groups = dataframe.groupby(hue)
    #for name, group in groups:
    #    axes.plot(group.x, group.y, marker='.', alpha=0.3, label=name)
    sns.scatterplot(data=dataset, x=dataset[:, 0], y=dataset[:, 1], hue=hue, ax=axes
                    , marker='.', alpha=0.3)
    axes.set_title(title)

    origin = [0, 0]
    vectors = []
    print(loadings)
    for j in range(7):
        vec = axes.quiver(*origin, loadings[j, 0], loadings[j, 1], color=colors[j], scale=5, zorder=100)
        vectors.append(vec)
    legend = axes.legend(vectors, labels, loc='upper left')
    axes.add_artist(legend)
    axes.set_xlim(-10, 35)
    axes.set_ylim(-15, 20)
    axes.axhline(0, color='black')
    axes.axvline(0, color='black')
    axes.legend(loc='best')

def PCALoadingsBeforeAfterPlots(df, title, day0):
    dataset2, hue2, loadings2, = PCAFitAndLoadingwithResampling(df.to_numpy()[:, 1:8], df.to_numpy()[:, 9])

    if day0:
        dataset3, hue3, loadings3, = PCAFitAndLoadings(df.to_numpy()[:, 1:8], df.to_numpy()[:, 10])
    else:
        dataset3, hue3, loadings3, = PCAFitAndLoadingwithResampling(df.to_numpy()[:, 1:8], df.to_numpy()[:, 10])

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    PCALoadingsBeforeAfterEachPlot(dataset2, hue2, loadings2, axes[0], "Before/After 2 Days")
    PCALoadingsBeforeAfterEachPlot(dataset3, hue3, loadings3, axes[1], "Before/After 3 Days")

    plt.suptitle(title)
    plt.savefig(title + ".jpg")
    plt.show()

def PCALoadingWithParametersForEachDataset(dataframe):

    dataset, hue, loadings = PCAFitAndLoadings(dataframe.to_numpy()[:, 1:8], dataframe.to_numpy()[:, 8])

    figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    index = [0, new_df1.shape[0], new_df1.shape[0] + new_df2.shape[0], dataframe.shape[0]]
    title = ["Core Collapse", "Type Ia", "Striped Envelope"]


    for k in range(3):
        PCALoadingsBeforeAfterEachPlot(dataset[index[k]:index[k + 1]], dataframe.to_numpy()[index[k]:index[k+1], 8],
                                       loadings, axes[0][k], title[k] + ": Supernova types")

        PCALoadingsBeforeAfterEachPlot(dataset[index[k]:index[k + 1], :], dataframe.to_numpy()[index[k]:index[k+1], 0],
                                       loadings, axes[1][k], title[k] + ": Days after explosion")

    plt.suptitle("PCA Loadings for each dataset with Parameters")
    plt.tight_layout()
    plt.savefig("PCA Loadings for each dataset with Parameters")
    plt.show()


def PCALoadingBeforeAfterNDaysForEachDataset(dataframe, figtitle, day0):
    dataset2, hue2, loadings2, = PCAFitAndLoadingwithResampling(dataframe.to_numpy()[:, 1:8], dataframe.to_numpy()[:, 9])

    if day0:
        dataset3, hue3, loadings3, = PCAFitAndLoadings(dataframe.to_numpy()[:, 1:8], dataframe.to_numpy()[:, 10])
    else:
        dataset3, hue3, loadings3, = PCAFitAndLoadingwithResampling(dataframe.to_numpy()[:, 1:8], dataframe.to_numpy()[:, 10])

    figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    index = [0, new_df1.shape[0], new_df1.shape[0] + new_df2.shape[0], dataframe.shape[0]]
    title = ["Core Collapse", "Type Ia", "Striped Envelope"]

    for k in range(3):
        PCALoadingsBeforeAfterEachPlot(dataframe[index[k]:index[k+1]], dataset2[index[k]:index[k+1]],
                                       hue2[index[k]:index[k+1]], loadings2, axes[0][k],
                                       title[k] + ": Before/After 2 days")

        PCALoadingsBeforeAfterEachPlot(dataframe[index[k]:index[k+1]], dataset3[index[k]:index[k+1], :],
                                       hue3[index[k]:index[k+1]], loadings3, axes[1][k],
                                       title[k] + ": Before/After 2 days")


    plt.suptitle(figtitle)
    plt.tight_layout()
    plt.savefig(figtitle + ".jpg")
    plt.show()


new_df1 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_CC_modified.csv", index_col=0)
new_df2 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_Ia_modified.csv", index_col=0)
new_df3 = pd.read_csv("/Users/athish/Coding/YSEProject/LCs_SESNe_modified.csv", index_col=0)

dataframes = [new_df1, new_df2, new_df3]
final_df = pd.concat([new_df1, new_df2, new_df3])
"""

for i in range(len(dataframes)):
    PCALoadingsPlotter(dataframes[i], titles[i], scalings1[i], i)


PCALoadingswithParametersPlot(final_df)
"""

#PCALoadingWithParametersForEachDataset(final_df)


for df in dataframes:
    Phasecut(df, 2)
    Phasecut(df, 3)

final_df_2 = pd.concat([new_df1, new_df2, new_df3])
print(final_df_2)
print(final_df_2.to_numpy()[:, 8])
PCALoadingsBeforeAfterPlots(final_df_2, "PCA Loadings before & after n days", True)

#PCALoadingBeforeAfterNDaysForEachDataset(final_df_2, "PCA Loadings before & after n days for each dataset", True)

new_df1 = new_df1.drop(np.arange(0, new_df1.shape[0], 6))
new_df2 = new_df2.drop(np.arange(0, new_df2.shape[0], 6))
new_df3 = new_df3.drop(np.arange(0, new_df3.shape[0], 6))
final_df_3 = pd.concat([new_df1, new_df2, new_df3])

#PCALoadingsBeforeAfterPlots(final_df_3, "PCA Loadings before & after n days without day 0 data", False)

#PCALoadingBeforeAfterNDaysForEachDataset(final_df_3, "PCA Loadings before & after n days for each dataset without day 0 data", False)
