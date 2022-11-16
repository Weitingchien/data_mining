from sklearn import preprocessing
import numpy as np  # 數學函式庫
import pandas as pd  # 處理數據
from sklearn.model_selection import cross_val_score  # 交叉驗證 用來調整超參數
import matplotlib.pyplot as plt
from sklearn import tree
import xgboost as xgb
from imblearn.over_sampling import SMOTE  # 用來作over-sampling的技術


pd.set_option('display.max_rows', None)  # print時 設置None顯示全部的資料

# 圓餅圖


def pie_charts(train_df):
    print(train_df['Type.1'].value_counts())
    features_counts = train_df['Type.1'].value_counts()
    plt.title('Type.1')
    # autopct顯示百分比值到小數點後第二位
    plt.pie(features_counts, labels=[
            val for val in features_counts], autopct='%.2f')
    plt.show()

# before()沒有作over-sampling


def train(train_X, train_y, test_X, test_y):
    # 交叉驗證(調整模型參數)
    for i in range(3, 20):
        # max_depth為樹的最大深度 、 random_state 設為 42 用來確保每次結果都相同
        clf = tree.DecisionTreeClassifier(max_depth=i, random_state=42)
        clf.fit(train_X, train_y)  # test dataset來訓練模型
        accuracy_test = clf.score(test_X, test_y)
        accuracy_train = clf.score(train_X, train_y)
        print(
            f"(DecisionTreeClassifier) accuracy of test dataset: {accuracy_test}、樹的最大深度: {i}")
        print(
            f"(DecisionTreeClassifier) accuracy of train dataset: {accuracy_train}、樹的最大深度: {i}")
        # max_depth為樹的最大深度，預設值: 6、 learning_rate(學習率)，預設:0.3、n_estimators(迭代次數)，預設:100
        clf = xgb.XGBClassifier(
            max_depth=i, learning_rate=0.1, n_estimators=100, random_state=42)
        clf.fit(train_X, train_y)
        accuracy_test = clf.score(test_X, test_y)
        accuracy_train = clf.score(train_X, train_y)
        print(
            f"(XGBClassifier) accuracy of test dataset: {accuracy_test}、樹的最大深度: {i}")
        print(
            f"(XGBClassifier) accuracy of train dataset: {accuracy_train}、樹的最大深度: {i}")


def before(train_X, train_y, test_df, test_X):
    test_y = test_df['Type.1']
    train(train_X, train_y, test_X, test_y)


# after()有作over-sampling
def after(train_df2, train_X, train_y, test_df, test_X):

    train_X, train_y = SMOTE(
        random_state=42, k_neighbors=2).fit_resample(train_df2, train_y)

    pie_charts(train_X)

    train_y = train_X['Type.1']
    test_y = test_df['Type.1']
    train_X = train_X.drop('Type.1', axis=1)

    train(train_X, train_y, test_X, test_y)


def main():
    train_df = pd.read_csv('pokemon_train.csv')
    test_df = pd.read_csv('pokemon_test.csv')
    train_df2 = train_df
    # preprocessing
    # 刪除第一個欄位(值是資料排序的數字,訓練模型時不需要這個值)
    train_df2 = train_df2.iloc[:, 1:]
    test_df = test_df.iloc[:, 1:]

    # FALSE -> 0, TRUE -> 1
    train_df2['Legendary'] = train_df2['Legendary'].astype(
        int)
    test_df['Legendary'] = test_df['Legendary'].astype(int)

    pie_charts(train_df2)

    # Type.2 缺值以None表示
    train_df2['Type.2'] = train_df2['Type.2'].fillna('None')
    test_df['Type.2'] = test_df['Type.2'].fillna('None')

    # 使用LabelEncoder()將字串轉換成數值
    LE = preprocessing.LabelEncoder()
    train_df2['Type.1'] = LE.fit_transform(train_df2['Type.1'])
    test_df['Type.1'] = LE.fit_transform(test_df['Type.1'])
    train_df2['Type.2'] = LE.fit_transform(train_df2['Type.2'])
    test_df['Type.2'] = LE.fit_transform(test_df['Type.2'])

    # Type.1要當成標籤(y)來訓練模型 所以這邊x刪除標籤
    train_X = train_df2.drop('Type.1', axis=1)
    train_y = train_df2['Type.1']
    test_X = test_df.drop('Type.1', axis=1)
    print(train_df2.head())
    before(train_X, train_y, test_df, test_X)
    after(train_df2, train_X, train_y, test_df, test_X)


main()
