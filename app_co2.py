import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_mars = pd.read_csv("mars-2014-complete.csv", sep = ';', encoding = 'cp850')

st.title("Projet Emission de CO2 par les véhicules")
st.sidebar.title("Sommaire")
pages=["Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
  st.write("### Introduction")

if page == pages[1] : 
  st.write("### DataVizualization")

# Sélection des colonnes de type float et int
Num_cols = df_mars.select_dtypes(include=['float64', 'int64']).columns

# Remplacement des valeurs float et int manquantes par la médiane de chaque colonne
df_mars[Num_cols] = df_mars[Num_cols].fillna(df_mars[Num_cols].median())

if page == pages[2] : 
  st.write("### Modélisation")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Supprimer les variables trop corellées avec la variable cible
df_mars = df_mars.drop(['conso_urb', 'conso_exurb', 'conso_mixte'], axis = 1)

# Réinitialiser les index après la suppression
df_mars = df_mars.reset_index(drop=True)

# Séparation du jeu de données sur la variable cible "co2"
X = df_mars.drop('co2', axis=1)
y = df_mars['co2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 42)

# Encodage des variables catégorielles
var_cat = ['lib_mrq', 'lib_mod_doss', 'lib_mod', 'dscom', 'cnit', 'tvv', 'cod_cbr', 'typ_boite_nb_rapp', 'Carrosserie', 'gamme']

label_encoder = LabelEncoder()
for column in var_cat:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.fit_transform(X_test[column])

# Normalisation des données
scaler = StandardScaler()
var_num = ['puiss_max','co_typ_1', 'hc', 'nox', 
               'hcnox', 'ptcl']
X_train.loc[:,var_num] = scaler.fit_transform(X_train[var_num])
X_test.loc[:,var_num] = scaler.transform(X_test[var_num])


# Instancier le modèle de régression par arbre de décision
from sklearn.tree import DecisionTreeRegressor 

regressor = DecisionTreeRegressor(random_state=42) 
  
regressor.fit(X_train, y_train)

print(regressor.score(X_train, y_train))
print(regressor.score(X_test ,y_test))

from sklearn.linear_model import LinearRegression

# Instanciation du modèle de régression linéaire
modele_regression_lineaire = LinearRegression()

# Entraînement du modèle sur les données d'entraînement
modele_regression_lineaire.fit(X_train, y_train)

# Évaluation du modèle
score_train = modele_regression_lineaire.score(X_train, y_train)
score_test = modele_regression_lineaire.score(X_test, y_test)

from sklearn.ensemble import RandomForestRegressor 
regressor = RandomForestRegressor(random_state=42) 
regressor.fit(X_train, y_train)
print(regressor.score(X_train, y_train))
print(regressor.score(X_test, y_test))

print("Score sur les données d'entraînement:", score_train)
print("Score sur les données de test:", score_test)

def prediction(classifier):
    if classifier == 'Random Forest':
        clf = RandomForestRegressor()
    elif classifier == 'Linear Regression':
        clf = LinearRegression()
    elif classifier == 'Decision Tree Regressor':
        clf = DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    return clf

choix = ['Random Forest', 'Linear Regression', 'Decision Tree Regressor']
option = st.selectbox('Choix du modèle', choix)
st.write('Le modèle choisi est :', option)