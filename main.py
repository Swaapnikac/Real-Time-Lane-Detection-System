import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('data.csv')
Y = df['SalePrice']
X = df.drop('SalePrice', axis = 1)
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.2)
rf = RandomForestRegressor(n_estimators = 10)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
print(r2_score(test_labels, predictions))
print(mean_squared_error(test_labels, predictions, squared=False))
Q = pd.read_csv("bpp_test.csv")
A = rf.predict(Q)
Q["prediction"] = A
filegen = Q[["prediction"]]
filegen.to_csv("submission.csv")
