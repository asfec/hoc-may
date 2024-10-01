import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

data = pd.read_csv('iris.csv')
x = data.drop('Species', axis=1)
y = data['Species']
# chia du lieu thanh 2 phan 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 42) 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(  acc)
