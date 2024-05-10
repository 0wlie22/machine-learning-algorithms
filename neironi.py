import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('winequality.csv', sep=';')


X = data.drop(columns=['quality'])  
y = data['quality']  


ratios = [0.8]

for exp_num, ratio in enumerate(ratios, start=1):

    train_mse_values = []
    test_mse_values = []
    

    for i in range(100):
   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - ratio), random_state=i)
   
        model = MLPRegressor(hidden_layer_sizes=(20,20), max_iter=800, random_state=i)
        model.fit(X_train, y_train)


        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        train_mse_values.append(train_mse)
        test_mse_values.append(test_mse)

    new_df = pd.DataFrame({'Apmācības datu kopās VKK': train_mse_values, 'Testa datu kopās VKK': test_mse_values})
    new_df.to_csv(f'experiment_{exp_num}_results.csv', index=False, sep=';')
    

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 101), train_mse_values, label='Apmācības datu kopās VKK', color='blue', linestyle='-')
    plt.plot(range(1, 101), test_mse_values, label='Testa datu kopās VKK', color='red', linestyle='-')
    plt.title(f'Eksperiments {exp_num}: Vidējā kvadrāta kļūda pret iterācijas skaitli')
    plt.xlabel('Iterācijas numurs')
    plt.ylabel('Vidējā kvadrāta kļūda')
    plt.legend()
    plt.grid(True)
    plt.show()
