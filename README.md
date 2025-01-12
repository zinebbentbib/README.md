# README.md
Ce projet analyse les données de ventes de smartphones et prédit les revenus futurs à l'aide de modèles de Machine Learning.
## 🚀 Fonctionnalités

-🔍Nettoyage et prétraitement des données
-🔍Analyse Exploratoire
Statistiques descriptives
Visualisation des données (histogrammes, boxplots, heatmap)
- 🤖 Modélisation
Régression via Random Forest
Évaluation du modèle (MSE, RMSE, R²)
-📊 Prédictions
Prédictions des revenus des smartphones pour les 3 premiers mois de 2025
Graphiques de comparaison des valeurs réelles et prédites


- telecom-sales-analysis/
├── data/                   #  Données
│   ├── raw             # 📥 Données brutes
│   └── processed/         # 📤 Données nettoyées et transformées
├── notebooks/              # 📓 Notebooks Jupyter pour l'exploration des données
│   └── exploration.ipynb
├── src/                    # 🛠️ Scripts Python (traitement et modélisation)
│   ├── data_processing.py  # Nettoyage et préparation des données
│   └── model.py   # Entraînement et évaluation du modèle (inclut tous le code)
├── README.md               # 📝 Présentation du projet
  
## ⚙️ Installation

1. **Cloner le dépôt GitHub :**

```bash
https://github.com/zinebbentbib/README.md.git
cd telecom-sales-analysis
pip install pandas matplotlib seaborn scikit-learn
