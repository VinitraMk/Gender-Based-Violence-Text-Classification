import pandas as pd

train = pd.read_csv('./input/Train.csv', encoding='ISO-8859-1')
test = pd.read_csv('./input/Test.csv', encoding = 'ISO-8859-1')
data = pd.concat([train, test])
data.to_csv('./input/data.csv', index = False)