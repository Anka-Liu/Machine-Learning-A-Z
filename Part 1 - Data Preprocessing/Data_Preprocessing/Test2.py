import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values=np.nan)
imputer_out = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

onehotencoder_X=OneHotEncoder(categorical_features=[0])
X=onehotencoder_X.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)