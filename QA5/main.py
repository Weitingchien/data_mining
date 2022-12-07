import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def display(x, y, title, ylabel='None'):
    plt.figure(figsize=(16, 8))
    plt.plot(x, y)
    plt.xlabel('K')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def elbow(K, inertias):
    K.append(len(inertias))
    ylabel = 'Distortion'
    title = 'Elbow method'
    display(K, inertias, title, ylabel)


def main():
    wine_dataset = pd.read_csv('wine.csv')

    attributes = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                  'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    """
    wine_dataset.columns = [index for index,
                            val in enumerate(wine_dataset.columns)]  # 添加column names(名稱即為索引)
    """

    wine_dataset.columns = [name for name in attributes]  # 添加欄位的名稱
    print(wine_dataset)
    labels = wine_dataset['Class'].values  # 取得有多少種酒 (資料集總共有3種酒)
    print(labels)

    X = wine_dataset.drop('Class', axis=1)  # 因k-means是unsupervised, 所以這邊把標籤刪除
    print(X)
    X = X.loc[:, :].values  # 取得所有欄位的值
    # print(X[:, 1])  # X[:, 1] 第一行的所有數據
    X = StandardScaler().fit_transform(X)  # 標準化
    pca = PCA(n_components=2)  # 使用PCA降維(維度: 2)
    pca_wine = pca.fit_transform(X)

    K = []
    inertias = []
    for k in range(1, 11):
        k_means = KMeans(n_clusters=k)
        k_means.fit(pca_wine)
        print(f"label: {k_means.labels_}")
        # 這邊因為沒有要預測新資料 所以直接拿訓練資料 主要是看分群之後的變化
        k_means_y = k_means.predict(pca_wine)
        plt.figure(figsize=(16, 8))
        plt.title(f'k-means: {k} groups')
        plt.scatter(pca_wine[:, 0], pca_wine[:, 1], c=k_means_y)
        # 找出每個cluster的cluster center(每個cluster都有一個centroid)
        centroid = k_means.cluster_centers_
        plt.scatter(centroid[:, 0], centroid[:, 1], c='red')
        inertias.append(k_means.inertia_)
        print(f"inertias: {inertias}")
        plt.show()

        elbow(K, inertias)


main()
