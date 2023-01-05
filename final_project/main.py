import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split

# 交叉驗證: 調整超參數


def cross_validation(pca_df, model='RF'):
    # print(pca_df)
    estimators = [50, 100, 150, 200, 250, 300, 350]  # 樹木的數量
    max_depth = [10, 20, 30, 40, 50, 60, 70]  # 最大深度
    # StratifiedKFold => 分層採樣、 n_splits表示數據要劃分幾份、shuffle是否打亂順序，設定成True會先打亂再劃分、random_state為0才會讓shuffle為True生效
    kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=0)
    scores = []

    if (model == 'RF'):
        # 交叉驗證找出n_estimators在多少棵樹木時會有最高準確率
        for index, n in enumerate(estimators):
            score = cross_val_score(RandomForestClassifier(
                random_state=0, n_estimators=n), pca_df.iloc[:, 0:2], pca_df['target'], cv=kf, scoring='accuracy')
            print(score)
            print(f"Average score:{score.mean()}")
            scores.append(score.mean())
        df = pd.DataFrame({'Value of estimators': estimators,
                          'cross-validation accuracy': scores})
        fig = px.line(df, x='Value of estimators',
                      y='cross-validation accuracy')
        fig.show()
        scores.clear()
        for index, n in enumerate(max_depth):
            score = cross_val_score(RandomForestClassifier(  # 先前已經交叉驗證過在沒有設定max_depth時，n_estimators在250準確率會是最高，所以這邊接下來找最佳max_depth的值
                random_state=0, n_estimators=250, max_depth=n), pca_df.iloc[:, 0:2], pca_df['target'], cv=kf, scoring='accuracy')
            print(score)
            print(f"Average score:{score.mean()}")
            scores.append(score.mean())
        df = pd.DataFrame({'Value of max_depth': max_depth,
                           'cross-validation accuracy': scores})
        fig = px.line(df, x='Value of max_depth',
                      y='cross-validation accuracy')
        fig.show()
        scores.clear()


def main():
    original_df = pd.read_excel('Rice_Osmancik_Cammeo_Dataset.xlsx')
    df = original_df
    df['CLASS'] = df['CLASS'].map({'Cammeo': 0, 'Osmancik': 1})

    # 檢查有無missing value
    print(df.isnull().sum())

    x = df.iloc[:, 0:7].values
    y = df['CLASS']
    # print(y)
    # 標準化
    X = StandardScaler().fit_transform(x)
    print(X)
    #print(np.mean(X), np.std(X))

    # print(pca_df)
    # 70%用來訓練模型 30%測試
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.3, random_state=0)  # train dataset: 2667  test dataset total: 1143

    pca = PCA(n_components=2)
    pca_train_X_cv = pca.fit_transform(X)
    pca_train_X = pca.fit_transform(train_X)
    pca_test_X = pca.fit_transform(test_X)
    pca_df = pd.DataFrame(data=pca_train_X_cv, columns=['PC1', 'PC2'])
    pca_df['target'] = y
    print(pca_df)
    # 隨機森林
    RF = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=0)
    RF.fit(pca_train_X, train_y)
    pred_y = RF.predict(pca_test_X)
    #print(f"feature importances{RF.feature_importances_}")
    print(f"Accuracy of train dataset(RF): {RF.score(pca_train_X, train_y)}")
    print(f"Accuracy of test dataset(RF): {RF.score(pca_test_X, test_y)}")
    print(classification_report(pred_y, test_y))
    # 進行交叉驗證
    cross_validation(pca_df)

    # 邏輯回歸
    LR = LogisticRegression(random_state=0)
    LR.fit(pca_train_X, train_y)
    print(f"Accuracy of train dataset(LR): {LR.score(pca_train_X, train_y)}")
    print(f"Accuracy of test dataset(LR): {LR.score(pca_test_X, test_y)}")
    pred_y = LR.predict(pca_test_X)

    # 混淆矩陣
    cm = confusion_matrix(test_y, pred_y)
    fig, px = plt.subplots(figsize=(10, 10))
    # 在空白圖中填充混淆矩陣
    px.matshow(cm, cmap=plt.cm.Greens)
    # shape[0]: rows shape[1]=> columns
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            px.text(x=i, y=j, s=cm[i, j])
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix', fontsize=15)
    plt.show()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 15)
    main()
