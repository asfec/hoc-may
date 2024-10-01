import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data =  pd.read_csv('knn-test.csv' , sep=';' ) 
data.columns = data.columns.str.strip()
x= data[['Feature 1','Feature 2']]
y= data['Label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

k = 1  
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác: {accuracy:.2f}")
