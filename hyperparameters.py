import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('UNSW_NB15.csv',sep=',').sample(2000)

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

def get_accuracy_score(individual):
    global X_train, y_train, X_test, y_test
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
        return abs(score)

def mutate(individual):
    # Kans op mutatie voor 'hidden_layer_sizes' 
    r = random.randint(1, 50)
    if r == 1:
        individual['hidden_layer_sizes'] = (random.randint(1, 5), random.randint(1, 5), random.randint(1, 5))
        print(f"Dit individu is gemuteerd en heeft een nieuwe waarde voor hidden layers: {individual['hidden_layer_sizes']}")
    
    # Kans op mutatie voor 'max_iter'
    r = random.randint(1, 50)
    if r == 1:
        individual['max_iter'] = random.randint(1, 5)
        print(f"Dit individu is gemuteerd en heeft een nieuwe waarde voor max iter: {individual['max_iter']}")

    # Kans op mutatie voor 'activation'
    r = random.randint(1, 50)
    if r == 1:
        individual['activation'] = random.choice(activation_functions)
        print(f"Dit individu is gemuteerd en heeft een nieuwe waarde voor de activatiefunctie: {individual['activation']}")

    # Kans op mutatie voor 'alpha'
    r = random.randint(1, 50)
    if r == 1:
        individual['alpha'] = round(random.uniform(0.001, 0.1), 3)
        print(f"Dit individu is gemuteerd en heeft een nieuwe waarde voor de alpha-waarde: {individual['alpha']}")

    # Kans op mutatie voor 'learning_rate_init'
    r = random.randint(1, 50)
    if r == 1:
        individual['learning_rate_init'] = round(random.uniform(0.001, 0.1), 3)
        print(f"Dit individu is gemuteerd en heeft een nieuwe waarde voor'learning_rate_init': {individual['learning_rate_init']}")

    return individual

# Initiële populatie genereren. Elke waarde is willekeurig
# De omvang van de getallen kan aangepast worden. (Dus (1, 50) kan omgezet worden naar (1, 100), bijvoorbeeld)
# Let op: een grote range kan er voor zorgen dat het algoritme erg lang duurt.
# Wanneer de ranges hier veranderd worden, moet dat ook in de mutatiefunctie (hierboven) gedaan worden!
population = []
activation_functions = ['relu', 'tanh', 'logistic', 'identity']
best_individuals = []
# Het getal in de range() functie bepaalt de grootte van de populatie
print("De initiële populatie wordt nu gegenereerd...")
for i in range(100):
    print(f"Individu {i} wordt gegenereerd.")
    population.append(
        {'hidden_layer_sizes': (random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)),
         'max_iter': random.randint(1, 5),
         'activation': random.choice(activation_functions),
         'alpha': round(random.uniform(0.001, 0.1), 3),
         'learning_rate_init': round(random.uniform(0.001, 0.1), 3)})

# Aantal generaties
for j in range(10):
    rankedpopulation = []
    for individual in population:
        rankedpopulation.append((fitness(individual), individual))
    
    print(rankedpopulation[0])
    print(rankedpopulation[1])
    rankedpopulation.sort(key=lambda x: x[0])
    rankedpopulation.reverse()

    print(f" === Generatie {j} beste individu === ")
    print(f"{rankedpopulation[0][1]} met een accuracy van {get_accuracy_score(rankedpopulation[0][1])}") 
    best_individuals.append(rankedpopulation[0])

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
        #print(f"Dit is de lijst met parameters: {parameters}")
        for i in range(len(bestpopulation)):
            values[j].append(bestpopulation[i][1][parameters[j]])
        #print(f"Dit is de lijst met values: {values}")
        j += 1

    # Een nieuw individu pakt voor elke hyperparameter een willekeurige waarde uit de lijst van alle waardes
    # voor de betreffende hyperparameter. De waardes uit die lijst zijn afkomstig van de individuen in 
    # de lijst 'bestpopulation'
    newPopulation = []
    for _ in range(100):
        newIndividual = {}
        for p in range(len(parameters)):
            newIndividual[parameters[p]] = random.choice(values[p])
        mutate(newIndividual) 
        newPopulation.append(newIndividual)
        print(f"Het individu {newIndividual} is toegevoegd aan de nieuwe generatie.")
    
    population = newPopulation

accuracies = []
for g in range((len(best_individuals))):
    accuracy = get_accuracy_score(best_individuals[g][1])
    accuracies.append(accuracy)
    
for h in range((len(best_individuals))):
    print(f"Het beste individu in generatie {h} is: {best_individuals[h][1]} met een accuracy van {accuracies[h]}.")
