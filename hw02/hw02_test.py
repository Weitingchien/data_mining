"""
from sklearn import tree
# orange: 0, apple: 1
features = [[150, 1], [170, 1], [130, 0], [140, 0]]

labels = [0, 0, 1, 1]

# 建立DecisionTreeClassifier模型
clf = tree.DecisionTreeClassifier()
# training set放入模型訓練
clf = clf.fit(features, labels)
predict1 = clf.predict([[120, 0]])
# 準確率
accuracy = clf.score(features, labels)
print(accuracy)
"""
from io import StringIO
import pydotplus
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
#from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.image as image
import matplotlib.pyplot as plt


def tree(times, df2, train_X, test_X, train_y, test_y, num):
    clf = DecisionTreeClassifier(
        criterion='entropy', max_leaf_nodes=num)  # 建立模型 max_leaf_nodes為最大葉節點個數
    clf.fit(train_X, train_y)  # 訓練模型
    pred_y = clf.predict(test_X)
    # print(f"正確率為: {accuracy_score(test_y, pred_y)}")
    if (accuracy_score(test_y, pred_y) == 1.0):
        display(times, df2, clf, train_X, test_X, train_y, test_y)


def display(times, df2, clf, train_X, test_X, train_y, test_y):
    times[0] += 1
    if (times[0] == 2):
        times[0] = 0
        return
    #print(f"Times: {times[0]}")
    io = StringIO()
    file_name = 'decision_tree.png'
    #columns_names = df2.columns
    columns_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    model = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=6)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    dot_data = export_graphviz(
        model, feature_names=columns_names, out_file=io, class_names=np.unique(train_y), filled=True)
    graph = pydotplus.graph_from_dot_data(io.getvalue())
    graph.write_png(file_name)
    img = image.imread(file_name)
    plt.figure(figsize=(100, 200))
    plt.imshow(img)


def main():
    times = [0]
    df = pd.read_csv('drug200.csv')  # 轉換程DataFrame 儲存到df變數
    # print(df.head())  # 印出前5筆資料
    """
    print(f"空值筆數: {df.isnull().values.sum()}")  # 檢查有無空值
    print(f"資料筆數: {df.shape}")
    print(f"資料的資料型別: {df.dtypes}")
    print(f"資料的欄位名稱: {df.keys()}")
    print(f"第一筆資料內容: {df.iloc[0,::]}")
    print(f"第一筆的預測目標: {df['Drug'][0]}")
    # 使用np.unique()取出欄位的值(值不重複)
    print(np.unique(df['Sex']))  # ['F' 'M']
    print(np.unique(df['BP']))  # ['HIGH' 'LOW' 'NORMAL']
    print(np.unique(df['Cholesterol']))  # ['HIGH' 'NORMAL']
    print(np.unique(df['Drug']))  # ['DrugY' 'drugA' 'drugB' 'drugC' 'drugX']
    """
    # 字串資料轉成數值, 才能開始進行訓練與預測
    df2 = df
    LE_sex = preprocessing.LabelEncoder()
    #LE_sex.fit(['F', 'M'])
    df2['Sex'] = LE_sex.fit_transform(df2['Sex'])
    LE_bp = preprocessing.LabelEncoder()
    df2['BP'] = LE_bp.fit_transform(df2['BP'])
    LE_cholesterol = preprocessing.LabelEncoder()
    df2['Cholesterol'] = LE_cholesterol.fit_transform(df2['Cholesterol'])
    y = df2['Drug']
    X = df2.drop(['Drug'], axis=1)
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.3, random_state=42)
    print(f"train_X: {train_X}")
    print(f"train_y: {train_y}")
    print(f"訓練集的維度大小: {train_X.shape}")
    print(f"測試集的維度大小: {test_X.shape}")
    print(train_y)
    print(df.head())
    for i in range(2, 10):
        tree(times, df2, train_X, test_X, train_y, test_y, i)  # 找出正確錄最高的最大葉節點個數


main()
