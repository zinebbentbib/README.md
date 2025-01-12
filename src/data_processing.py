# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 12:10:11 2025

@author: zineb
"""
import pandas as pd
# Charger le fichier CSV
file_path ="C:/Users/zineb/OneDrive/Bureau/Data/raw.csv"
data = pd.read_csv(file_path)

#  Nettoyage des données ---
# Vérifier les valeurs manquantes: 
print("Valeurs manquantes avant nettoyage :")
print(data.isnull().sum())

# Remplir les valeurs numériques manquantes avec la moyenne
columns_to_fill_mean = ['marketing_score', 'competition_index', 'customer_satisfaction', 
                        'purchasing_power_index', 'store_traffic']
data[columns_to_fill_mean] = data[columns_to_fill_mean].fillna(data[columns_to_fill_mean].mean())

# Remplir les valeurs catégoriques manquantes avec la modalité la plus fréquente
columns_to_fill_mode = ['weather_condition', 'tech_event', '5g_phase', 'public_transport']
for col in columns_to_fill_mode:
    data[col] = data[col].fillna(data[col].mode()[0])

# Supprimer les lignes avec des valeurs critiques manquantes dans les colonnes des revenus
data = data.dropna(subset=['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue'])

# Convertir la colonne des dates en type datetime
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])

# Vérifier les valeurs manquantes après nettoyage
print("\nValeurs manquantes après nettoyage :")
print(data.isnull().sum())


