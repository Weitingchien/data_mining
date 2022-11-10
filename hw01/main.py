import mplcursors  # 提供互動式資料游標
import numpy as np  # 數學函式庫
import pandas as pd  # 處理數據
import matplotlib.pyplot as plt  # 繪圖
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # 平均與變異數標準化

# PCA存在的問題: 數據維度變小之後, 每一維的物理含意便無法解釋


def display(original_df, pca_df):
    plt.figure(figsize=(15, 15))  # figsize(a,b)設置圖形大小, a是圖形的寬, b為圖形的高
    # 返回一個資料類型為Series(類似陣列的物件)：陣列內容為為原始數據整體診斷為0的索引值
    normal_series = original_df[original_df['OVERALL_DIAGNOSIS'] == 0].index
    # print(normal_series)
    abnormal_series = original_df[original_df['OVERALL_DIAGNOSIS'] == 1].index
    # 轉成串列(list)

    normal = list(normal_series)
    abnormal = list(abnormal_series)
    # 使用loc(row, column)可以指定行, 列數據
    # 取得attribute為'PC1', 值為0(無心臟病)的資料,
    normal_a = pca_df.loc[normal, ['PC1']]
    # print(normal_a)
    normal_b = pca_df.loc[normal, ['PC2']]

    abnormal_a = pca_df.loc[abnormal, ['PC1']]
    abnormal_b = pca_df.loc[abnormal, ['PC2']]
    # scatter(x,y)為散佈圖
    plt.scatter(normal_a, normal_b, c='b')
    plt.scatter(abnormal_a, abnormal_b, c='r')

    plt.title('PCA of SPECTF Heart Dataset', fontsize=25)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(['0', '1'])
    mplcursors.cursor(hover=True)
    plt.show()

# 心臟單個質子發射計算機斷層顯像(SPECT)的圖像數據


def main():
    FXR = ['OVERALL_DIAGNOSIS']  # 0=>normal(無心臟病) 1=> abnormal(有心臟病)
    for i in range(1, 45):
        if i % 2 == 0:
            FXR.append(f'F{i}S')
        else:
            FXR.append(f'F{i}R')
    # 建立1為陣列 值為數據集的各個特徵
    names = np.array(FXR)
    # print(names)
    original_df = pd.read_csv(
        'SPECTF.csv', names=names)  # 187 rows * 45 columns
    print(original_df.shape)

    df = original_df.drop(labels='OVERALL_DIAGNOSIS',
                          axis=1)  # PCA 非監督式學習降維方法 把標籤拿掉
    # print(df)
    # 取得F1R~F44S行的值
    x = df.loc[:, :].values
    # print(x)
    # 盡量將資料轉化為均值為0, 變異數(variance)為1: 使某個attribute的值不會因為數值特別大影響整個計算
    # X為標準化後的數據
    X = StandardScaler().fit_transform(x)
    # print(X)
    print(np.mean(X), np.std(X))  # mean:平均數, std: 標準差
    normalised_SPECTF_heart = pd.DataFrame(X)
    # print(normalised_SPECTF_heart)
    pca = PCA(n_components=2)  # n_components所要保留的特徵數量(降維之後的維度為2)
    # fit_transform: 求數據x的均值, 變異數, 最大,最小值, 再對數據進行轉換, 來實現標準化
    pca__SPECTF_heart = pca.fit_transform(X)
    # print(pca.explained_variance_ratio_)  # 特徵維度的變異數比例
    # print(pca.explained_variance_)  # 特徵維度的變異數: 數據若是要區分開來, 離散程度就需要比較大(變異數大)
    pca_df = pd.DataFrame(data=pca__SPECTF_heart, columns=['PC1', 'PC2'])

    display(original_df, pca_df)


main()
