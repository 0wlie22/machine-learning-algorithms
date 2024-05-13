import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('winequality.csv', sep=';')

X = data.drop(columns=['quality'])
y = data['quality']

neighbors_settings = [10, 50, 100]

for exp_num, k_neighbors in enumerate(neighbors_settings, start=1):
    train_acc_values = []
    test_acc_values = []

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - 0.8), random_state=i)

        model = KNeighborsClassifier(k_neighbors, metric='euclidean')
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        train_acc_values.append(train_acc)
        test_acc_values.append(test_acc)

    new_df = pd.DataFrame({'Train Accuracy': train_acc_values, 'Test Accuracy': test_acc_values})
    new_df.to_csv(f'experiment_{exp_num}_results.csv', index=False, sep=';')

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 101), train_acc_values, label='Apmācības datu precizitāte', color='blue', linestyle='-')
    plt.plot(range(1, 101), test_acc_values, label='Testa datu precizitāte', color='red', linestyle='-')
    plt.title(f'Eksperiments {exp_num}: Precizitāte pret Iterācijas Numuru ar k_neighbors: {k_neighbors}')
    plt.xlabel('Iterācijas numurs')
    plt.ylabel('Precizitāte')
    plt.legend()
    plt.grid(True)
    plt.show()
