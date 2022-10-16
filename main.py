import os
import csv
import random as rd
import numpy as np
import pandas as pd
import mplcursors
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_breast_cancer, load_iris  # 引入 dataset
from sklearn.preprocessing import StandardScaler  # 平均與變異數標準化
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def display(original_df, pca_df):
    plt.figure(figsize=(15, 15))  # figsize(a,b)設置圖形大小, a是圖形的寬, b為圖形的高
    # zip(labels, colors) => [('<=50K', 'r'),( '>50K', 'b')]
    less_than_or_equal_to_50K_series = original_df[original_df['label']
                                                   == ' <=50K'].index
    greater_than_50K_series = original_df[original_df['label']
                                          == ' >50K'].index
    less_than_or_equal_to_50K = list(less_than_or_equal_to_50K_series)
    greater_than_50K = list(greater_than_50K_series)

    less_than_or_equal_to_50K_a = pca_df.loc[less_than_or_equal_to_50K, [
        'PC1']].PC1
    less_than_or_equal_to_50K_b = pca_df.loc[less_than_or_equal_to_50K, [
        'PC2']].PC2

    greater_than_50K_a = pca_df.loc[greater_than_50K, ['PC1']].PC1
    greater_than_50K_b = pca_df.loc[greater_than_50K, ['PC2']].PC2
    plt.scatter(less_than_or_equal_to_50K_a,
                less_than_or_equal_to_50K_b, c='r')
    plt.scatter(greater_than_50K_a, greater_than_50K_b, c='b')

    plt.title('PCA of Adult Census Income Dataset', fontsize=25)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend([' <=50K', ' >50K'], prop={'size': 15})
    mplcursors.cursor(hover=True)
    plt.show()


def main():  # indexcol=False 不使用第一列作為索引
    names = np.array(['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                      'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label'])

    original_df = pd.read_csv('adult.csv', index_col=False,
                              names=names)  # 32560 rows * 15 colums
    # original_df['label'].replace(str.strip())
    # 把字串清除, 才標準化
    filter_names = np.array(['workclass', 'education', 'marital-status', 'sex',
                             'native-country', 'occupation', 'relationship', 'race', 'label'])

    df = original_df.drop(labels=filter_names, axis=1)
    # setdiff1d() 回傳names與filter_names的差集
    differenceSet = list(np.setdiff1d(names, filter_names))
    # print(df)
    x = df.loc[:, differenceSet].values
    # 盡量將資料轉化為均值為0, 變異數(variance)為1
    # X為標準化後的數據
    X = StandardScaler().fit_transform(x)
    print(np.mean(X), np.std(X))
    normalised_adult_census_income = pd.DataFrame(
        X, columns=differenceSet)
    # print(normalised_adult)
    pca = PCA(n_components=2)  # (n_components所要保留的特徵數量)
    pca__adult_census_income = pca.fit_transform(X)
    # print(pca.explained_variance_ratio_)
    pca_df = pd.DataFrame(data=pca__adult_census_income,
                          columns=['PC1', 'PC2'])

    display(original_df, pca_df)


main()
