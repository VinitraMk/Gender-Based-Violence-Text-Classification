import pandas as pd

train = pd.read_csv('./input/Train.csv')
test = pd.read_csv('./input/Test.csv')
data = pd.concat([train, test])
data.to_csv('./input/data.csv')