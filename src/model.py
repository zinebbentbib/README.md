
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np           
from sklearn.metrics import mean_absolute_error
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



# Statistiques descriptives pour les revenus des 3 modèles de smartphones
print("\nStatistiques descriptives des revenus :")
desc_stats = data[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']].describe()
print(desc_stats)

# Visualisation des revenus journaliers ---
#Extraire les colonnes des revenus
revenues = data[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']]

#Visualisation des revenus journaliers
plt.figure(figsize=(14, 7))
revenues.plot(title="Revenus Journaliers des Smartphones")
plt.ylabel("Revenus (€)")
plt.xlabel("Index")
plt.grid(True)
plt.show()


# Boxplots avec échelle logarithmique
plt.figure(figsize=(14, 7))
data[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']].boxplot()
plt.yscale('log')
plt.title("Boxplots des Revenus par Modèle de Smartphone (Échelle Logarithmique)", fontsize=16)
plt.ylabel("Revenus (€) (Échelle Log)")
plt.xticks([1, 2, 3], ['jPhone Pro', 'Kaggle Pixel 5', 'Planet SX'])
plt.grid(True)
plt.show()

# Histogrammes pour observer les distributions
data[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']].hist(bins=30, figsize=(14, 7))
plt.suptitle("Distributions des Revenus par Modèle de Smartphone")
plt.show()

# Calcul des corrélations
correlation_matrix = data[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue',
                           'marketing_score', 'competition_index', 'customer_satisfaction',
                           'purchasing_power_index', 'store_traffic']].corr()

# Affichage de la matrice de corrélation
print("\nMatrice de corrélation :")
print(correlation_matrix)

# Visualisation des corrélations avec une heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Corrélations entre les revenus et les variables exogènes")
plt.show()

# Relations spécifiques (scatterplots)
# Exemple 1 : Relation entre marketing_score et les revenus
plt.figure(figsize=(12, 6))
sns.scatterplot(x='marketing_score', y='jPhone_Pro_revenue', data=data, label="jPhone Pro")
sns.scatterplot(x='marketing_score', y='Kaggle_Pixel_5_revenue', data=data, label="Kaggle Pixel 5")
sns.scatterplot(x='marketing_score', y='Planet_SX_revenue', data=data, label="Planet SX")
plt.title("Impact du Marketing Score sur les Revenus")
plt.xlabel("Marketing Score")
plt.ylabel("Revenus (€)")
plt.legend()
plt.grid(True)
plt.show()

# Exemple 2 : Relation entre satisfaction client et revenus
plt.figure(figsize=(12, 6))
sns.lineplot(x='customer_satisfaction', y='jPhone_Pro_revenue', data=data, label="jPhone Pro")
sns.lineplot(x='customer_satisfaction', y='Kaggle_Pixel_5_revenue', data=data, label="Kaggle Pixel 5")
sns.lineplot(x='customer_satisfaction', y='Planet_SX_revenue', data=data, label="Planet SX")
plt.title("Impact de la Satisfaction Client sur les Revenus")
plt.xlabel("Customer Satisfaction")
plt.ylabel("Revenus (€)")
plt.legend()
plt.grid(True)
plt.show()


# Analyse locale par ville
city_revenue = data.groupby('city')[['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']].mean()
city_revenue = city_revenue.sort_values(by='jPhone_Pro_revenue', ascending=False)
plt.figure(figsize=(16, 8))
city_revenue.plot(kind='bar', figsize=(16, 8), grid=True)
plt.title("Revenus Moyens par Ville et Modèle de Smartphone", fontsize=16)
plt.ylabel("Revenus Moyens (€)")
plt.xlabel("Ville")
plt.xticks(rotation=45, fontsize=10)
plt.legend(title="Modèles", fontsize=10)
plt.show()

# Étape 3 : Modèle de prévision
label_encoder = LabelEncoder()
categorical_columns = ['weather_condition', 'tech_event', '5g_phase', 'public_transport', 'city']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

features = ['marketing_score', 'competition_index', 'customer_satisfaction',
            'purchasing_power_index', 'store_traffic', 'weather_condition',
            'tech_event', '5g_phase', 'public_transport', 'city']
targets = ['jPhone_Pro_revenue', 'Kaggle_Pixel_5_revenue', 'Planet_SX_revenue']

X = data[features]
y = data[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Étape 4 : Prédictions pour 2025
future_dates = pd.date_range(start='2025-01-01', end='2025-03-31', freq='D')
future_data = pd.DataFrame({
    'date': future_dates,
    'marketing_score': np.random.uniform(50, 100, len(future_dates)),
    'competition_index': np.random.uniform(1, 10, len(future_dates)),
    'customer_satisfaction': np.random.uniform(70, 100, len(future_dates)),
    'purchasing_power_index': np.random.uniform(50, 120, len(future_dates)),
    'store_traffic': np.random.uniform(100, 1000, len(future_dates)),
    'weather_condition': np.random.choice(data['weather_condition'].unique(), len(future_dates)),
    'tech_event': np.random.choice(data['tech_event'].unique(), len(future_dates)),
    '5g_phase': np.random.choice(data['5g_phase'].unique(), len(future_dates)),
    'public_transport': np.random.choice(data['public_transport'].unique(), len(future_dates)),
    'city': np.random.choice(data['city'].unique(), len(future_dates))
})

future_predictions = model.predict(future_data[features])
future_data[targets] = future_predictions
print(future_data.head())

# Sauvegarder les prédictions
future_data.to_csv("C:/Users/zineb/Documents/predictions_2025.csv", index=False)
print("Prédictions sauvegardées dans 'predictions_2025.csv'.") 


# 1️⃣ Calcul des erreurs pour chaque modèle
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Erreur Absolue Moyenne (MAE): {mae:.2f}")
print(f"Erreur Quadratique Moyenne (MSE): {mse:.2f}")
print(f"Racine de l'Erreur Quadratique Moyenne (RMSE): {rmse:.2f}")

# 2️⃣ Courbes Réel vs Prédit pour chaque modèle
for i, target in enumerate(targets):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.iloc[:, i].values[:100], label='Valeurs Réelles', marker='o', linestyle='--')
    plt.plot(y_pred[:, i][:100], label='Valeurs Prédites', marker='x', linestyle='-')
    plt.title(f"Prédictions vs Réel - {target}")
    plt.xlabel("Observations")
    plt.ylabel("Revenus (€)")
    plt.legend()
    plt.grid(True)
    plt.show()

# 3️⃣ Matrice des erreurs résiduelles
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 6))
sns.heatmap(residuals, cmap='coolwarm', cbar=True, center=0)
plt.title("Matrice des Résidus")
plt.xlabel("Modèles de Smartphones")
plt.ylabel("Observations")
plt.show()

# 4️⃣ Distribution des résidus
plt.figure(figsize=(10, 6))
sns.histplot(residuals.flatten(), bins=30, kde=True, color='purple')
plt.title("Distribution des Résidus")
plt.xlabel("Erreur (Résidus)")
plt.ylabel("Fréquence")
plt.grid(True)
plt.show()

# 5️⃣ Importance des variables
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Importance des Variables dans le Modèle")
plt.xlabel("Importance")
plt.ylabel("Variables")
plt.grid(True)
plt.show()

