from sklearn import tree
from sklearn import preprocessing
import numpy as np  # 數學函式庫
import pandas as pd  # 處理數據


def main():
    training_df = pd.read_csv('pokemon_train.csv')
    training_df2 = training_df
    LE = preprocessing.LabelEncoder()  # 使用LabelEncoder()將字串轉換成數值
    # ['Bug' 'Dark' 'Dragon' 'Electric' 'Fairy' 'Fighting' 'Fire' 'Flying' 'Ghost' 'Grass' 'Ground' 'Ice' 'Normal' 'Poison' 'Psychic' 'Rock' 'Steel' 'Water']
    # print(np.unique(training_df2['Type.1']))
    training_df2['Type.1'] = LE.fit_transform(training_df2['Type.1'])
    training_df2['Type.2'] = LE.fit_transform(training_df2['Type.2'])
    training_df2['Legendary'] = LE.fit_transform(
        training_df2['Legendary'])
    X = training_df2.drop(['Type.1', 'Type.2'], axis=1)
    print(X)
    # print(np.unique(training_df2['Type.1']))
    # print(training_df)


main()
