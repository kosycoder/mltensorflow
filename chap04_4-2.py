import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.DataFrame([
    ['green', 'M', '10.1', 'class2'],
    ['red', 'L', '13.5', 'class1'],
    ['blue', 'XL', '15.3', 'class2']
])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)
print(df)
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'] = df['size'].map(inv_size_mapping)
print(df)

class_mapping = {label: idx for idx, label in
                 enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([('onehot', OneHotEncoder(), [0]),
                              ('nothing', 'passthrough', [1, 2])])
print(c_transf.fit_transform(X).astype(float))
