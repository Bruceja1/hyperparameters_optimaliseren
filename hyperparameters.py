# https://www.educative.io/answers/implement-neural-network-for-classification-using-scikit-learn

import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from sklearn.preprocessing import StandardScaler

# dataset = load_digits()
dataset = pd.read_csv('UNSW_NB15.csv',sep=',')
# print(dataset.head())
# print(dataset.info())
# print(dataset.isnull().sum())
# print(dataset['label'].value_counts())

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

def get_accuracy_score(individual):
    global X_train, y_train, X_test, y_test
    convertedValues = convert_values_to_string(individual)
    # mlpc = MLPClassifier(hidden_layer_sizes=[50,50,50], max_iter=50, alpha=0.001, activation='identity', learning_rate_init=0.1,)
    mlpc = MLPClassifier(convertedValues)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)

    print(classification_report(y_test, pred_mlpc))
    print(confusion_matrix(y_test, pred_mlpc))

    cm = accuracy_score(y_test, pred_mlpc)
    print(cm)
    return cm
    
def fitness(individual):
    score = get_accuracy_score(individual)

    if score == 1:
        return 99999
    else:
        return abs(1/score)

# Methode om de gegenereerde hyperparameters omzetten naar een string die als parameter in de MLPClassifier() 
# methode gezet kan worden.    
def convert_values_to_string(individual):
    convertedValues = ""
    keyList = list(individual.keys())
    valueList = list(individual.values())
    
    for i in range(len(keyList)):
        if isinstance(valueList[i], str):
            convertedValues += f"{keyList[i]} = '{valueList[i]}', "
        else:
            convertedValues += f"{keyList[i]} = {valueList[i]}, "

    print(f"The converted values are: {convertedValues}")
    return convertedValues


# Initiële populatie (hyperparameters) genereren. Elke waarde is willekeurig
# De omvang van de getallen kan aangepast worden. (Dus 1, 50 kan omgezet worden naar 1, 100, bijvoorbeeld)
population = []
activation_functions = ['relu', 'tanh', 'logistic', 'identity']
# Het getal in de range() functie bepaalt de grootte van de populatie
for i in range(100):
    population.append(
        {'hidden_layer_sizes': (random.randint(1, 50), random.randint(1, 50), random.randint(1, 50)),
         'max_iter': random.randint(1, 1000),
         'activation': random.choice(activation_functions),
         'alpha': random.uniform(0.001, 0.1),
         'learning_rate_init': random.uniform(0.001, 0.1)})

for i in range(10):
    rankedpopulation = []
    for s in population:
        rankedpopulation.append((fitness(s), s))
    rankedpopulation.sort()
    rankedpopulation.reverse()

    print(f" === Gen {i} best solutions === ")
    print(rankedpopulation[0]) 

""" print(population[0])
print(population[0].keys())
print(population[0].values()) """
