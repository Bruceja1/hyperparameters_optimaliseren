# https://www.educative.io/answers/implement-neural-network-for-classification-using-scikit-learn

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from sklearn.preprocessing import StandardScaler

# dataset = load_digits()
dataset = pd.read_csv('UNSW_NB15.csv',sep=',')
print(dataset.head())
print(dataset.info())
# print(dataset.isnull().sum())
print(dataset['label'].value_counts())

# https://pythonprogramming.net/working-with-non-numerical-data-machine-learning-tutorial/
def handle_non_numerical_data(dataset):
    columns = dataset.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
            column_contents = dataset[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            
            dataset[column] = list(map(convert_to_int, dataset[column]))

    return dataset

dataset = handle_non_numerical_data(dataset)
print(dataset.head())

X = dataset.drop('label', axis = 1)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#print(X_train[:10])

# https://www.youtube.com/watch?v=0Lt9w-BxKFQ
mlpc = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)

print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))

cm = accuracy_score(y_test, pred_mlpc)
print(cm)
