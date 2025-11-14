
# Segmentation K-means du dataset *Wine Quality* (White)

Ce projet met en œuvre une **segmentation non supervisée** des vins blancs portugais (*Wine Quality – Vinho Verde, UCI*) à l’aide de **K-means** en Python, à travers :

- Un **notebook d’analyse complète** : `wine-quality.ipynb` (exploration, prétraitement, choix de k, interprétation).
- Une **application Streamlit interactive** : `streamlit_app.py` (exploration des données, sélection de k, entraînement, visualisation des clusters, prédiction pour de nouveaux vins).

L’ensemble est conçu pour un contexte académique de **machine learning non supervisé**, avec une structuration professionnelle et des bonnes pratiques reproductibles.

---

## 1. Objectifs du projet

- Explorer les **caractéristiques physico-chimiques** des vins blancs.
- Appliquer **K-means** pour regrouper les vins en profils homogènes.
- Évaluer la qualité des clusters à l’aide de **mesures internes**.
- Étudier les liens entre clusters et **qualité perçue** (`quality`) sans utiliser cette variable dans l’entraînement.
- Fournir une **interface utilisateur** permettant :
  - de configurer les paramètres de K-means,
  - d’entraîner un modèle,
  - de visualiser les résultats,
  - de prédire le cluster d’un nouveau vin à partir de ses caractéristiques.

---

## 2. Données

- Dataset : `winequality-white.csv` (UCI Machine Learning Repository).
- Format :
  - séparateur `;`
  - point `.` comme séparateur décimal.

Chaque ligne représente un vin, avec :

**Variables explicatives (11 variables continues)**  
`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`,  
`free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`.

**Variable de qualité**  
`quality` : note entière (0–10), **non utilisée pour l’apprentissage de K-means**, uniquement pour l’analyse des clusters.

---

## 3. Structure du projet

```bash
├── wine-quality.ipynb        # Notebook d'analyse et de clustering K-means
├── streamlit_app.py          # Application Streamlit interactive
├── winequality-white.csv     # Dataset UCI (vins blancs, séparateur ;)
├── kmeans_model.joblib       # Modèle K-means sauvegardé
├── scaler.joblib             # StandardScaler sauvegardé
├── kmeans_model.pkl          # Sauvegarde alternative
├── scaler.pkl                
├── model_metadata.pkl        # Métadonnées du modèle (K choisi, features, etc.)
└── README.md                 # Documentation du projet
```

Les fichiers de modèles (`*.joblib`, `*.pkl`) et de métadonnées sont générés automatiquement par l’application Streamlit lors de l’entraînement.

---

## 4. Prérequis & installation

### 4.1. Environnement

- **Python** : 3.9 ou plus récent recommandé.
- OS : Windows / Linux / macOS.

### 4.2. Installation des dépendances (en trois étapes: création de l'environnement virtuel .venv, activation de l'environnement virtuel, installation des dépendances)

N.B: Les bibliothèques utilisés dans le projet sont : numpy pandas scikit-learn matplotlib seaborn plotly joblib streamlit

Dans le dossier `Projet_k-means` :

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# ou
.venv\Scripts\activate       # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Utilisation du notebook (`wine-quality.ipynb`)

Le notebook permet de reproduire en détail le pipeline K-means.

### 5.1. Lancer le notebook

Depuis la racine du projet :

```bash
jupyter notebook
```

Puis ouvrez le fichier `wine-quality.ipynb` et exécutez les cellules dans l’ordre.

### 5.2. Contenu typique

Le notebook illustre les étapes suivantes :

1. **Chargement & description des données**
   - Aperçu du dataset, `info()`, statistiques descriptives.
2. **Prétraitement**
   - Gestion des valeurs manquantes (si nécessaire).
   - Suppression d’éventuels doublons.
   - Standardisation des variables explicatives à l’aide de `StandardScaler`.
3. **Choix du nombre de clusters K**
   - Méthode du coude (inertie).
   - Score de silhouette.
   - Autres indices internes (Calinski–Harabasz, Davies–Bouldin).
4. **Entraînement du modèle K-means**
   - Entraînement avec le K retenu.
5. **Interprétation des clusters**
   - Visualisation 2D via PCA.
   - Analyse de la distribution de `quality` par cluster.
   - Profils moyens des variables par cluster.
6. **Sauvegarde du modèle (optionnel)**
   - Export du `StandardScaler` et du modèle K-means pour réutilisation dans l’application Streamlit.

---

## 6. Application Streamlit (`streamlit_app.py`)

L’application Streamlit propose une interface graphique pour :

- charger les données,
- explorer le dataset,
- configurer les paramètres de K-means,
- rechercher un K optimal,
- entraîner le modèle final,
- visualiser et interpréter les clusters,
- prédire le cluster d’un nouveau vin.

### 6.1. Lancement de l’application

Depuis la racine du projet :

```bash
streamlit run streamlit_app.py
```

Conditions :

- `winequality-white.csv` doit être présent à la racine du projet.
- Les bibliothèques nécessaires doivent être installées.

### 6.2. Fonctionnalités principales

L’interface inclut :

1. **Page d’accueil / Introduction**
   - Présentation du contexte, des objectifs et du jeu de données.

2. **Exploration des données**
   - Aperçu du dataset.
   - Statistiques descriptives.
   - Visualisation des distributions des variables.
   - Matrice de corrélation, focus sur le lien entre variables physico-chimiques et `quality`.

3. **Configuration de K-means**
   - Choix des paramètres :
     - Intervalle de recherche pour K (`k_min`, `k_max`).
     - `n_init`, `max_iter`, `random_state`, `tol`, etc.
   - Calcul automatique de plusieurs métriques internes :
     - Inertie (méthode du coude),
     - Score de silhouette,
     - Indice de Davies–Bouldin,
     - Indice de Calinski–Harabasz.
   - Proposition d’un **K recommandé** à partir des métriques.

4. **Entraînement**
   - Entraînement du modèle K-means avec les paramètres retenus.
   - Affichage des scores obtenus pour le K choisi.

5. **Visualisation des résultats**
   - Répartition des observations par cluster.
   - Graphiques :
     - PCA 2D colorée par cluster,
     - boxplots ou heatmaps des moyennes par cluster,
     - Comparaisons `cluster vs quality`.
   - Analyse détaillée des profils de chaque cluster.

6. **Prédiction pour un nouveau vin**
   - Formulaire interactif permettant de saisir les 11 caractéristiques physico-chimiques.
   - Application du `StandardScaler` entraîné, puis du modèle K-means.
   - Retour :
     - du **cluster prédit**,
     - d’informations sur le profil typique de ce cluster,
     - de visualisations comparatives

---

## 7. Méthodologie (synthèse)

Principes clés du projet :

- **Standardisation obligatoire** des variables avant K-means.
- **Recherche rigoureuse de K** :
  - croisement de plusieurs critères (coude, silhouette, indices internes),
  - possibilité d’évaluer plusieurs K dans l’application.
- **Interprétation centrée métier** :
  - lecture des clusters en termes de profils physico-chimiques,
  - mise en relation éventuelle avec la qualité (`quality`) pour donner du sens aux groupes.
- **Reproductibilité** :
  - séparation claire entre analyse (notebook) et interface (Streamlit),
  - sauvegarde des modèles et métadonnées,
  - configuration contrôlée (`random_state`).

---

## 8. Pistes d’amélioration

Quelques prolongements possibles :

- Tester d’autres algorithmes de clustering :
  - `MiniBatchKMeans`, `GaussianMixture`, `DBSCAN`, clustering hiérarchique, etc.
- Ajouter des options avancées :
  - gestion plus poussée des outliers,
  - sélection automatique des features les plus pertinentes.
- Étendre le projet :
  - intégration du dataset des vins rouges,
  - comparaison entre différents modèles et différents types de vins.
- Intégrer un déploiement :
  - conteneurisation avec Docker,
  - déploiement sur un serveur ou une plateforme cloud.

---

## 9. Licence & crédits

- Projet réalisé dans un cadre pédagogique autour du **machine learning non supervisé** et du clustering K-means.
- Données issues du **UCI Machine Learning Repository – Wine Quality (Vinho Verde)**.
- Licence : **MIT**

