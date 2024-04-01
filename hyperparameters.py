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
    # convertedValues = convert_values_to_string(individual)
    # mlpc = MLPClassifier(hidden_layer_sizes=[50,50,50], max_iter=50, alpha=0.001, activation='identity', learning_rate_init=0.1,)
    mlpc = MLPClassifier(**individual)
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
""" def convert_values_to_string(individual):
    convertedValues = ""
    keyList = list(individual.keys())
    valueList = list(individual.values())
    
    for i in range(len(keyList)):
        if isinstance(valueList[i], str):
            convertedValues += f"{keyList[i]}='{valueList[i]}', "
        else:
            convertedValues += f"{keyList[i]}={valueList[i]}, "

    print(f"The converted values are: {convertedValues}")
    return convertedValues """


# Initiële populatie (hyperparameters) genereren. Elke waarde is willekeurig
# De omvang van de getallen kan aangepast worden. (Dus (1, 50) kan omgezet worden naar (1, 100), bijvoorbeeld)
# Let op: een grote range kan er voor zorgen dat het algoritme erg lang duurt.
population = []
activation_functions = ['relu', 'tanh', 'logistic', 'identity']
# Het getal in de range() functie bepaalt de grootte van de populatie
for i in range(20):
    population.append(
        {'hidden_layer_sizes': (random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)),
         'max_iter': random.randint(1, 5),
         'activation': random.choice(activation_functions),
         'alpha': random.uniform(0.001, 0.1),
         'learning_rate_init': random.uniform(0.001, 0.1)})

for j in range(5):
    rankedpopulation = []
    for individual in population:
        rankedpopulation.append((fitness(individual), individual))
    
    print(rankedpopulation[0])
    print(rankedpopulation[1])
    rankedpopulation.sort(key=lambda x: x[0])
    rankedpopulation.reverse()

    print(f" === Gen {j} best solutions === ")
    print(rankedpopulation[0]) 

    bestpopulation = rankedpopulation[:10]

    # Bevat alle namen van de hyperparameters
    parameters = []
    # Elke index i binnen de value lijst bevat een lijst van alle waardes van de hyperparameter op index i in 
    # parameters[]. Deze lijst wordt gebruikt om de waardes van de hyperparameters van de nieuwe generatie
    # te selecteren.
    values = []
    j = 0
    for p in population[0].keys():
        parameters.append(p)
        values.append([])
        print(f"Dit is de lijst met parameters: {parameters}")
        for i in range(len(bestpopulation)):
            values[j].append(bestpopulation[i][1][parameters[j]])
        print(f"Dit is de lijst met values: {values}")
        j += 1

    # Een nieuw individu pakt voor elke hyperparameter een willekeurige waarde uit de lijst van alle waardes
    # voor de betreffende hyperparameter.
    newPopulation = []
    for _ in range(20):
        newIndividual = {}
        for p in range(len(parameters)):
            newIndividual[parameters[p]] = random.choice(values[p])
        # mutatie hier 
        newPopulation.append(newIndividual)
        print(f"Het individu {newIndividual} is toegevoegd aan de nieuwe generatie.")
    
    print(parameters)
    print(values)
    print(newPopulation[0])
    rankedpopulation = newPopulation
   





""" print(population[0])
print(population[0].keys())
print(population[0].values()) """

# DEBUGGING
""" 

mlpc = MLPClassifier(hidden_layer_sizes=(50,50,50), max_iter=50, alpha=0.001, activation='identity', learning_rate_init=0.1, )
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)

print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))

cm = accuracy_score(y_test, pred_mlpc)
print(cm)
"""