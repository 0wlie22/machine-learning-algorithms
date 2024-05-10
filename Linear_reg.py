import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('winequality.csv', sep=';')


X = data.drop(columns=['quality']) 
y = data['quality'] 


ratios = [0.5, 0.6, 0.7]


for exp_num, ratio in enumerate(ratios, start=1):

    train_mse_values = []
    test_mse_values = []
    

    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - ratio), random_state=i)
        

        model = LinearRegression()
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        train_mse_values.append(train_mse)
        test_mse_values.append(test_mse)

    new_df = pd.DataFrame({'Train MSE': train_mse_values, 'Test MSE': test_mse_values})
    new_df.to_csv(f'experiment_{exp_num}_results.csv', index=False, sep=';')
    

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 101), train_mse_values, label='apmācības datu kopās Vidējā kvadrāta kļūda', color='blue', linestyle='-')
    plt.plot(range(1, 101), test_mse_values, label='testa datu kopās Vidējā kvadrāta kļūda', color='red', linestyle='-')
    plt.title(f'Eksoeriments {exp_num}: Vidējā kvadrāta kļūda vs Iteracijas Numurs')
    plt.xlabel('Iteratcijas numurs')
    plt.ylabel('Vidējā kvadrāta kļūda')
    plt.legend()
    plt.grid(True)
    plt.show()
