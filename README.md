# README.md
Ce projet analyse les données de ventes de smartphones et prédit les revenus futurs à l'aide de modèles de Machine Learning.
## 🚀 Fonctionnalités

- 🔍 Nettoyage et prétraitement des données
- 📈 Analyses statistiques et visualisations
- 🤖 Modélisation prédictive avec Random Forest
- 📊 Prédictions des revenus pour l'année 2025

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
git clone https://github.com/ton-utilisateur/telecom-sales-analysis.git
cd telecom-sales-analysis
pip install pandas matplotlib seaborn scikit-learn
