import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Read dataset
data = pd.read_csv("winequality.csv", delimiter=";")

data["color"] = data["color"].astype("category")
# Transform color column from "red/white" to 0 and 1
data["color_cat"] = data["color"].cat.codes

# We are interested only in these attributes
reduced_data = data[["residual_sugar", "chlorides", "density", "sulphates", "alcohol", "color_cat"]]

# Normalize data
# reduced_data["fixed_acidity"] = (reduced_data["fixed_acidity"] - reduced_data["fixed_acidity"].min()) / (reduced_data["fixed_acidity"].max() - reduced_data["fixed_acidity"].min())
# reduced_data["volatile_acidity"] = (reduced_data["volatile_acidity"] - reduced_data["volatile_acidity"].min()) / (reduced_data["volatile_acidity"].max() - reduced_data["volatile_acidity"].min())
# reduced_data["citric_acid"] = (reduced_data["citric_acid"] - reduced_data["citric_acid"].min()) / (reduced_data["citric_acid"].max() - reduced_data["citric_acid"].min())
reduced_data["residual_sugar"] = (reduced_data["residual_sugar"] - reduced_data["residual_sugar"].min()) / (reduced_data["residual_sugar"].max() - reduced_data["residual_sugar"].min())
reduced_data["chlorides"] = (reduced_data["chlorides"] - reduced_data["chlorides"].min()) / (reduced_data["chlorides"].max() - reduced_data["chlorides"].min())
# reduced_data["free_sulfur_dioxide"] = (reduced_data["free_sulfur_dioxide"] - reduced_data["free_sulfur_dioxide"].min()) / (reduced_data["free_sulfur_dioxide"].max() - reduced_data["free_sulfur_dioxide"].min())
# reduced_data["total_sulfur_dioxide"] = (reduced_data["total_sulfur_dioxide"] - reduced_data["total_sulfur_dioxide"].min()) / (reduced_data["total_sulfur_dioxide"].max() - reduced_data["total_sulfur_dioxide"].min())
reduced_data["density"] = (reduced_data["density"] - reduced_data["density"].min()) / (reduced_data["density"].max() - reduced_data["density"].min())
# reduced_data["ph"] = (reduced_data["ph"] - reduced_data["ph"].min()) / (reduced_data["ph"].max() - reduced_data["ph"].min())
reduced_data["sulphates"] = (reduced_data["sulphates"] - reduced_data["sulphates"].min()) / (reduced_data["sulphates"].max() - reduced_data["sulphates"].min())
reduced_data["alcohol"] = (reduced_data["alcohol"] - reduced_data["alcohol"].min()) / (reduced_data["alcohol"].max() - reduced_data["alcohol"].min())

# Selecting features for clustering
cluster_data = reduced_data[["alcohol", "residual_sugar", "chlorides", "sulphates"]]

# Selecting features for classification
reduced_data = data[['color_cat'] + [col for col in data.columns if col != 'color' and col != 'color_cat']]

# Define the target variable and split proportion
izejas = reduced_data['color_cat']
apmacibas_datu_proporcija = 0.7

# Split the dataset into training and test sets
X_apmacibas, X_testa, y_apmacibas, y_testa = train_test_split(
    cluster_data, izejas, test_size=1.0 - apmacibas_datu_proporcija, random_state=42
)

# Calculate counts and percentages for training data
apmacibas_datu_skaits = y_apmacibas.value_counts()
apmacibas_datu_procenti = y_apmacibas.value_counts(normalize=True) * 100

# Calculate counts and percentages for test data
testa_datu_skaits = y_testa.value_counts()
testa_datu_procenti = y_testa.value_counts(normalize=True) * 100

# Print training data statistics
print("Datu objektu skaits apmācības datu kopā:", len(y_apmacibas))
print("Datu objektu % proporcija apmācības datu kopā:")
for klase in apmacibas_datu_skaits.index:
    print(f"Klase {klase}: {apmacibas_datu_skaits[klase]} ({apmacibas_datu_procenti[klase]:.2f}%)")

# Print test data statistics
print("\nDatu objektu skaits testa datu kopā:", len(y_testa))
print("Datu objektu % proporcija testa datu kopā:")
for klase in testa_datu_skaits.index:
    print(f"Klase {klase}: {testa_datu_skaits[klase]} ({testa_datu_procenti[klase]:.2f}%)")

# KNN Experiments
neighbors_list = [5, 10, 50]
for n_neighbors in neighbors_list:
    print(f"\nExperiment with n_neighbors = {n_neighbors}")

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_apmacibas, y_apmacibas)

    y_pred_knn = knn_model.predict(X_testa)

    print(classification_report(y_testa, y_pred_knn, zero_division=0))

    cm = confusion_matrix(y_testa, y_pred_knn)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Prognozētas klašu iezīmes')
    ax.set_ylabel('Īstās klašu iezīmes')
    ax.set_title(f'Kļūdu matrica (n_neighbors = {n_neighbors})')
    ax.xaxis.set_ticklabels(data['color'].unique())
    ax.yaxis.set_ticklabels(data['color'].unique())
    plt.show()
