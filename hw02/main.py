from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
#from sklearn.metrics import accuracy_score
import numpy as np  # 數學函式庫
import pandas as pd  # 處理數據
# from sklearn.preprocessing import MultiLabelBinarizer  # 平均與變異數標準化
from sklearn.model_selection import cross_val_score  # 交叉驗證 用來調整超參數


pd.set_option('display.max_rows', None)  # print時 設置None顯示全部的資料


def main():
    train_df = pd.read_csv('pokemon_train.csv')
    # print(train_df.info())
    # print(train_df.isnull().sum())
    test_df = pd.read_csv('pokemon_test.csv')
    #valid_df = pd.read_csv('pokemon_valid.csv')
    train_df2 = train_df
    train_df2 = train_df2.iloc[:, 1:]  # 刪除第一個欄位(值是資料排序的數字,訓練模型時不需要這個值)
    test_df = test_df.iloc[:, 1:]
    # preprocessing
    #train_df2['Type.2'] = train_df2['Type.2'].fillna('None')
    # print(train_df2['Type.2'])
    #test_df['Type.2'] = test_df['Type.2'].fillna('None')

    LE = preprocessing.LabelEncoder()  # 使用LabelEncoder()將字串轉換成數值
    # ['Bug' 'Dark' 'Dragon' 'Electric' 'Fairy' 'Fighting' 'Fire' 'Flying' 'Ghost' 'Grass' 'Ground' 'Ice' 'Normal' 'Poison' 'Psychic' 'Rock' 'Steel' 'Water']
    # print(np.unique(train_df2['Type.1']))
    train_df2['Type.1'] = LE.fit_transform(train_df2['Type.1'])
    train_df2['Type.2'] = LE.fit_transform(train_df2['Type.2'])
    test_df['Type.1'] = LE.fit_transform(test_df['Type.1'])
    test_df['Type.2'] = LE.fit_transform(test_df['Type.2'])
    # print(train_df2['Type.2'])

    train_X = train_df2.drop(
        ['Type.1', 'Total', 'Generation', 'Legendary'], axis=1)
    test_X = test_df.drop(
        ['Type.1', 'Total', 'Generation', 'Legendary'], axis=1)

    train_y = train_df2['Type.1']
    test_y = test_df['Type.1']
    for i in range(3, 20):
        clf = tree.DecisionTreeClassifier(max_depth=i, random_state=42)
        clf.fit(train_X, train_y)  # test dataset來訓練模型
        scores = cross_val_score(clf, train_X, train_y, cv=5)
        print(scores)
        # print(clf.feature_importances_)
        accuracy_test = clf.score(test_X, test_y)
        accuracy_train = clf.score(train_X, train_y)
        print(accuracy_test)
        print(accuracy_train)
    """
    random_forest_model = RandomForestClassifier(n_estimators=10000)
    random_forest_model.fit(X, train_y)
    random_forest_predicted = random_forest_model.predict(test_X)
    accuracy_random_forest_test = random_forest_model.score(test_X, test_y)
    """
    # print(random_forest_predicted)
    # print(accuracy_random_forest_test)


main()
