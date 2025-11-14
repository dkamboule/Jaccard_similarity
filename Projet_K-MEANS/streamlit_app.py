"""
Application Streamlit pour l'analyse K-means du dataset Wine Quality
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import KFold
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from typing import Optional, Dict, Any
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse du dataset wine quality à l'aide de l'algorithme des k-means",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B0082;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
def initialize_session_state():
    """Initialise tous les états de session nécessaires"""
    default_states = {
        'page': 'Introduction',
        'model_trained': False,
        'kmeans_model': None,
        'scaler': None,
        'clustered_data': None,
        'config': None,
        'final_k': None,
        'X_scaled': None,
        'data_loaded': False,
        'df': None,
        'features_selected': [],
        'k_range': range(2, 11)
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Fonction de chargement des données
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Charge et nettoie le dataset avec gestion robuste des erreurs"""
    try:
        # Chargement des données depuis le contenu fourni
        data = """fixed acidity;volatile acidity;citric acid;residual sugar;chlorides;free sulfur dioxide;total sulfur dioxide;density;pH;sulphates;alcohol;quality
7;0.27;0.36;20.7;0.045;45;170;1.001;3;0.45;8.8;6
6.3;0.3;0.34;1.6;0.049;14;132;0.994;3.3;0.49;9.5;6
8.1;0.28;0.4;6.9;0.05;30;97;0.9951;3.26;0.44;10.1;6"""
        
        # Utiliser les données fournies dans la question
        from io import StringIO
        df = pd.read_csv(StringIO(data), sep=';')
        
        st.session_state.data_loaded = True
        st.session_state.df = df
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erreur critique lors du chargement des données : {e}")
        return None

# Fonction de validation croisée
def cross_validate_kmeans(X: np.ndarray, k_range: range, n_splits: int = 5) -> Dict[int, Dict[str, float]]:
    """Validation croisée pour sélectionner le meilleur k"""
    try:
        if len(X) < n_splits:
            n_splits = max(2, len(X) // 2)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {k: {'silhouette': [], 'davies_bouldin': [], 'calinski': []} for k in k_range}
        
        progress_bar = st.progress(0)
        total_iterations = len(k_range) * n_splits
        current_iteration = 0
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_train)
                    
                    if len(X_val) > 0:
                        val_labels = kmeans.predict(X_val)
                        
                        if len(np.unique(val_labels)) > 1:
                            results[k]['silhouette'].append(silhouette_score(X_val, val_labels))
                            results[k]['davies_bouldin'].append(davies_bouldin_score(X_val, val_labels))
                            results[k]['calinski'].append(calinski_harabasz_score(X_val, val_labels))
                    
                except Exception as e:
                    continue
                
                current_iteration += 1
                if total_iterations > 0:
                    progress_bar.progress(min(current_iteration / total_iterations, 1.0))
        
        progress_bar.empty()
        
        # Calcul des moyennes
        cv_results = {}
        for k in k_range:
            cv_results[k] = {
                'silhouette_mean': np.mean(results[k]['silhouette']) if results[k]['silhouette'] else -1,
                'silhouette_std': np.std(results[k]['silhouette']) if results[k]['silhouette'] else 0,
                'davies_bouldin_mean': np.mean(results[k]['davies_bouldin']) if results[k]['davies_bouldin'] else float('inf'),
                'calinski_mean': np.mean(results[k]['calinski']) if results[k]['calinski'] else 0
            }
        
        return cv_results
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la validation croisée : {e}")
        return {}

# Menu de navigation
def sidebar_navigation():
    """Menu de navigation dans la sidebar"""
    st.sidebar.title("🍷 Navigation")
    pages = ['Introduction', 'Exploration des données', 'Configuration k-means', 
             'Entrainement', 'Résultats', 'Prédiction']
    
    selected_page = st.sidebar.radio("Choisir une page", pages)
    st.session_state.page = selected_page
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 État de l'application")
    
    # État des données
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Données chargées")
        if st.session_state.df is not None:
            st.sidebar.info(f"📊 {len(st.session_state.df)} échantillons")
    else:
        st.sidebar.warning("📥 Données non chargées")
    
    # État du modèle
    if st.session_state.model_trained:
        st.sidebar.success("✅ Modèle entraîné")
        if st.session_state.final_k:
            st.sidebar.info(f"🎯 {st.session_state.final_k} clusters")
    else:
        st.sidebar.info("⏳ Modèle non entraîné")
    
    return selected_page

# Fonction utilitaire pour afficher les erreurs
def display_error(message: str, details: str = ""):
    """Affiche un message d'erreur formaté"""
    st.markdown(f"""
    <div class="error-box">
        <h4>❌ {message}</h4>
        <p>{details}</p>
    </div>
    """, unsafe_allow_html=True)

# Page d'introduction
def page_introduction():
    """Page d'introduction du projet"""
    st.markdown('<h1 class="main-header">🍷 Analyse k-means du dataset wine quality</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Description du projet
    st.markdown("## 📋 Description du projet")
    st.info("""
    Cette application interactive permet d'analyser le dataset **wine quality** en utilisant 
    l'algorithme de clustering **k-means**. L'objectif est de segmenter les vins en groupes 
    homogènes basés sur leurs caractéristiques physico-chimiques.
    """)
    
    # Vérification des données
    st.markdown("## 🔍 Vérification des données")
    
    if st.button("🔄 Charger les données", type="primary"):
        with st.spinner("Chargement des données..."):
            df = load_data()
            if df is not None:
                st.success(f"✅ Données chargées avec succès : {len(df)} échantillons, {len(df.columns)} variables")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Échantillons", len(df))
                with col2:
                    st.metric("Variables", len(df.columns))
                with col3:
                    st.metric("Dimensions", f"{len(df)} × {len(df.columns)}")
                
                # Aperçu des données
                st.markdown("### 📋 Aperçu des données")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Description statistique
                st.markdown("### 📊 Statistiques descriptives")
                st.dataframe(df.describe(), use_container_width=True)
            else:
                display_error("Échec du chargement des données")

# Page d'exploration des données
def page_exploration():
    """Page d'exploration des données"""
    st.markdown('<h1 class="main-header">📊 Exploration des données</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        display_error("Données non chargées", 
                     "Veuillez d'abord charger les données depuis la page d'introduction.")
        return
    
    df = st.session_state.df
    if df is None or df.empty:
        display_error("Données non disponibles")
        return
    
    try:
        # Onglets pour l'exploration
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Structure", "📈 Distributions", "🔍 Corrélations", "🎯 Target Analysis"])
        
        with tab1:
            st.markdown("### Structure des données")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Informations générales**")
                st.write(f"- **Nombre d'observations** : {len(df)}")
                st.write(f"- **Nombre de variables** : {len(df.columns)}")
                st.write(f"- **Variables numériques** : {len(df.select_dtypes(include=[np.number]).columns)}")
                
            with col2:
                st.markdown("**Types de données**")
                type_counts = df.dtypes.value_counts()
                for dtype, count in type_counts.items():
                    st.write(f"- {dtype} : {count}")
            
            st.markdown("**Aperçu des données**")
            st.dataframe(df, use_container_width=True)
            
            st.markdown("**Résumé statistique**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            st.markdown("### Distributions des variables")
            
            # Sélection de la variable à visualiser
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_var = st.selectbox("Choisir une variable", numeric_cols)
            
            if selected_var:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[selected_var].hist(bins=30, ax=ax, alpha=0.7)
                    ax.set_title(f'Distribution de {selected_var}')
                    ax.set_xlabel(selected_var)
                    ax.set_ylabel('Fréquence')
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    df[selected_var].plot(kind='box', ax=ax)
                    ax.set_title(f'Boxplot de {selected_var}')
                    st.pyplot(fig)
        
        with tab3:
            st.markdown("### Matrice de corrélation")
            
            # Calcul de la matrice de corrélation
            corr_matrix = df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax,
                       square=True, fmt='.2f', linewidths=0.5)
            ax.set_title('Matrice de Corrélation des Variables du Vin')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            st.pyplot(fig)
            
            # Corrélations avec la qualité
            st.markdown("**Corrélations avec la qualité du vin**")
            quality_corr = corr_matrix['quality'].sort_values(ascending=False)
            st.dataframe(quality_corr, use_container_width=True)
        
        with tab4:
            st.markdown("### Analyse de la variable cible (Quality)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                df['quality'].value_counts().sort_index().plot(kind='bar', ax=ax)
                ax.set_title('Distribution des Notes de Qualité')
                ax.set_xlabel('Qualité')
                ax.set_ylabel('Nombre de vins')
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                # Relation entre alcohol et quality
                fig, ax = plt.subplots(figsize=(10, 6))
                df.boxplot(column='alcohol', by='quality', ax=ax)
                ax.set_title('Teneur en Alcool par Niveau de Qualité')
                ax.set_xlabel('Qualité')
                ax.set_ylabel('Alcool (%)')
                st.pyplot(fig)
                
    except Exception as e:
        display_error("Erreur lors de l'exploration des données", str(e))

# Page de configuration
def page_configuration():
    """Page de configuration des paramètres k-means"""
    st.markdown('<h1 class="main-header">⚙️ Configuration k-means</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        display_error("Données non chargées", 
                     "Veuillez d'abord charger les données depuis la page d'introduction.")
        return
    
    df = st.session_state.df
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Sélection des caractéristiques")
            
            # Sélection des features
            available_features = [col for col in df.columns if col != 'quality']
            selected_features = st.multiselect(
                "Sélectionnez les variables pour le clustering:",
                available_features,
                default=available_features[:5] if len(available_features) > 5 else available_features
            )
            
            if not selected_features:
                st.warning("⚠️ Veuillez sélectionner au moins une variable")
                return
            
            st.session_state.features_selected = selected_features
            
            st.markdown("### 📊 Aperçu des données sélectionnées")
            st.dataframe(df[selected_features].head(), use_container_width=True)
        
        with col2:
            st.markdown("### ⚙️ Paramètres du clustering")
            
            # Paramètre k
            min_k = st.slider("Nombre minimum de clusters (k_min)", 2, 10, 2)
            max_k = st.slider("Nombre maximum de clusters (k_max)", 3, 15, 8)
            
            if min_k >= max_k:
                st.error("❌ k_min doit être inférieur à k_max")
                return
            
            k_range = range(min_k, max_k + 1)
            st.session_state.k_range = k_range
            
            # Paramètres avancés
            st.markdown("### 🔧 Paramètres avancés")
            n_init = st.slider("Nombre d'initialisations", 5, 20, 10)
            max_iter = st.slider("Nombre maximum d'itérations", 100, 500, 300)
            random_state = st.number_input("Random state", 0, 100, 42)
            
            # Normalisation des données
            normalize = st.checkbox("Normaliser les données", value=True)
            
        # Configuration finale
        st.markdown("### 💾 Configuration finale")
        
        config = {
            'features': selected_features,
            'k_range': k_range,
            'n_init': n_init,
            'max_iter': max_iter,
            'random_state': random_state,
            'normalize': normalize
        }
        
        st.session_state.config = config
        
        # Affichage récapitulatif
        st.success("✅ Configuration sauvegardée")
        st.json(config)
        
        # Prévisualisation des données préparées
        st.markdown("### 🔄 Préparation des données")
        X = df[selected_features]
        
        if normalize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state.scaler = scaler
            st.session_state.X_scaled = X_scaled
            st.info("📏 Données normalisées (StandardScaler)")
        else:
            st.session_state.X_scaled = X.values
            st.info("📊 Données brutes (sans normalisation)")
        
        st.write(f"**Shape des données préparées** : {st.session_state.X_scaled.shape}")
        
    except Exception as e:
        display_error("Erreur lors de la configuration", str(e))

# Page d'entraînement
def page_training():
    """Page d'entraînement du modèle"""
    st.markdown('<h1 class="main-header">🎓 Entrainement du modèle</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        display_error("Données non chargées")
        return
    
    if 'config' not in st.session_state or st.session_state.config is None:
        display_error("Configuration manquante", 
                     "Veuillez d'abord configurer les paramètres dans la page 'Configuration k-means'.")
        return
    
    try:
        X_scaled = st.session_state.X_scaled
        config = st.session_state.config
        k_range = config['k_range']
        
        st.markdown("### 🔍 Validation croisée")
        
        if st.button("🚀 Lancer l'entraînement et la validation", type="primary"):
            with st.spinner("Validation croisée en cours..."):
                # Validation croisée
                cv_results = cross_validate_kmeans(X_scaled, k_range)
                
                if not cv_results:
                    display_error("Échec de la validation croisée")
                    return
                
                # Analyse des résultats
                st.markdown("### 📊 Résultats de la validation croisée")
                
                # Préparation des données pour visualisation
                k_values = list(cv_results.keys())
                silhouette_scores = [cv_results[k]['silhouette_mean'] for k in k_values]
                db_scores = [cv_results[k]['davies_bouldin_mean'] for k in k_values]
                calinski_scores = [cv_results[k]['calinski_mean'] for k in k_values]
                
                # Trouver le meilleur k basé sur silhouette
                best_k_silhouette = k_values[np.argmax(silhouette_scores)]
                best_k_db = k_values[np.argmin(db_scores)]
                
                # Affichage des métriques
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Meilleur k (Silhouette)", best_k_silhouette)
                with col2:
                    st.metric("Meilleur k (Davies-Bouldin)", best_k_db)
                with col3:
                    st.metric("k optimal choisi", best_k_silhouette)
                
                # Visualisation des métriques
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Score de silhouette
                axes[0, 0].plot(k_values, silhouette_scores, 'bo-', linewidth=2, markersize=8)
                axes[0, 0].set_title('Score de Silhouette vs Nombre de Clusters')
                axes[0, 0].set_xlabel('Nombre de clusters (k)')
                axes[0, 0].set_ylabel('Score de Silhouette')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Score de Davies-Bouldin
                axes[0, 1].plot(k_values, db_scores, 'ro-', linewidth=2, markersize=8)
                axes[0, 1].set_title('Score de Davies-Bouldin vs Nombre de Clusters')
                axes[0, 1].set_xlabel('Nombre de clusters (k)')
                axes[0, 1].set_ylabel('Score de Davies-Bouldin')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Score de Calinski-Harabasz
                axes[1, 0].plot(k_values, calinski_scores, 'go-', linewidth=2, markersize=8)
                axes[1, 0].set_title('Score de Calinski-Harabasz vs Nombre de Clusters')
                axes[1, 0].set_xlabel('Nombre de clusters (k)')
                axes[1, 0].set_ylabel('Score de Calinski-Harabasz')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Score combiné
                axes[1, 1].plot(k_values, silhouette_scores, 'bo-', label='Silhouette', linewidth=2)
                axes[1, 1].plot(k_values, [1/x if x != 0 else 0 for x in db_scores], 'ro-', label='1/Davies-Bouldin', linewidth=2)
                axes[1, 1].set_title('Métriques Combinées')
                axes[1, 1].set_xlabel('Nombre de clusters (k)')
                axes[1, 1].set_ylabel('Score normalisé')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Entraînement final avec le meilleur k
                st.markdown("### 🎯 Entraînement du modèle final")
                
                final_k = best_k_silhouette
                st.session_state.final_k = final_k
                
                kmeans_final = KMeans(
                    n_clusters=final_k,
                    random_state=config['random_state'],
                    n_init=config['n_init'],
                    max_iter=config['max_iter']
                )
                
                kmeans_final.fit(X_scaled)
                st.session_state.kmeans_model = kmeans_final
                
                # Prédictions
                labels = kmeans_final.predict(X_scaled)
                st.session_state.clustered_data = st.session_state.df.copy()
                st.session_state.clustered_data['cluster'] = labels
                
                st.session_state.model_trained = True
                
                st.success(f"✅ Modèle entraîné avec succès! {final_k} clusters créés.")
                
                # Métriques finales
                st.markdown("### 📈 Métriques finales")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    silhouette_final = silhouette_score(X_scaled, labels)
                    st.metric("Silhouette Score", f"{silhouette_final:.3f}")
                
                with col2:
                    db_final = davies_bouldin_score(X_scaled, labels)
                    st.metric("Davies-Bouldin", f"{db_final:.3f}")
                
                with col3:
                    calinski_final = calinski_harabasz_score(X_scaled, labels)
                    st.metric("Calinski-Harabasz", f"{calinski_final:.1f}")
        
    except Exception as e:
        display_error("Erreur lors de l'entraînement", str(e))

# Page de résultats
def page_results():
    """Page d'affichage des résultats"""
    st.markdown('<h1 class="main-header">📊 Résultats de lanalyse</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        display_error("Modèle non entraîné", 
                     "Veuillez d'abord entraîner le modèle dans la page 'Entrainement'.")
        return
    
    try:
        clustered_data = st.session_state.clustered_data
        kmeans_model = st.session_state.kmeans_model
        final_k = st.session_state.final_k
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Clusters", "📊 Visualisation", "📈 Analyse", "💾 Export"])
        
        with tab1:
            st.markdown("### 📋 Répartition des clusters")
            
            # Distribution des clusters
            cluster_dist = clustered_data['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                cluster_dist.plot(kind='bar', ax=ax, color='skyblue', alpha=0.7)
                ax.set_title('Distribution des Clusters')
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Nombre d\'échantillons')
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 8))
                cluster_dist.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title('Proportion des Clusters')
                ax.set_ylabel('')
                st.pyplot(fig)
            
            # Données clusterisées
            st.markdown("### 📊 Données avec clusters")
            st.dataframe(clustered_data, use_container_width=True)
        
        with tab2:
            st.markdown("### 🎨 Visualisation des clusters")
            
            # Réduction de dimension avec PCA
            X_scaled = st.session_state.X_scaled
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            clustered_data['PCA1'] = X_pca[:, 0]
            clustered_data['PCA2'] = X_pca[:, 1]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=clustered_data['cluster'], 
                               cmap='viridis', alpha=0.6)
            ax.set_title('Visualisation 2D des Clusters (PCA)')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
            
            st.info(f"📊 La PCA explique {pca.explained_variance_ratio_.sum():.2%} de la variance totale")
        
        with tab3:
            st.markdown("### 📈 Analyse des clusters")
            
            # Caractéristiques par cluster
            numeric_cols = clustered_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'cluster']
            
            cluster_means = clustered_data.groupby('cluster')[numeric_cols].mean()
            
            st.markdown("#### 📊 Moyennes par cluster")
            st.dataframe(cluster_means.style.background_gradient(cmap='Blues'), 
                        use_container_width=True)
            
            # Heatmap des différences
            st.markdown("#### 🔥 Heatmap des caractéristiques par cluster")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', center=0, 
                       ax=ax, fmt='.2f', linewidths=0.5)
            ax.set_title('Caractéristiques Moyennes par Cluster')
            st.pyplot(fig)
        
        with tab4:
            st.markdown("### 💾 Export des résultats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export des données clusterisées
                csv = clustered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger les données clusterisées (CSV)",
                    data=csv,
                    file_name="wine_clusters.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export du modèle
                if st.button("💾 Sauvegarder le modèle"):
                    model_data = {
                        'model': st.session_state.kmeans_model,
                        'scaler': st.session_state.scaler,
                        'features': st.session_state.config['features'],
                        'k': st.session_state.final_k
                    }
                    
                    joblib.dump(model_data, 'kmeans_wine_model.pkl')
                    st.success("✅ Modèle sauvegardé comme 'kmeans_wine_model.pkl'")
            
            # Résumé exécutif
            st.markdown("### 📋 Résumé exécutif")
            st.info(f"""
            **Résumé de l'analyse :**
            - 🎯 **Nombre de clusters** : {final_k}
            - 📊 **Échantillons analysés** : {len(clustered_data)}
            - 📏 **Variables utilisées** : {len(st.session_state.config['features'])}
            - ⚡ **Performance** : Silhouette score de {silhouette_score(X_scaled, clustered_data['cluster']):.3f}
            - 🎨 **Clusters distincts** : {len(clustered_data['cluster'].unique())}
            """)
                
    except Exception as e:
        display_error("Erreur lors de l'affichage des résultats", str(e))

# Page de prédiction
def page_prediction():
    """Page de prédiction pour nouveaux échantillons"""
    st.markdown('<h1 class="main-header">🔮 Prédiction de cluster</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        display_error("Modèle non entraîné")
        return
    
    try:
        model = st.session_state.kmeans_model
        scaler = st.session_state.scaler
        features = st.session_state.config['features']
        
        st.markdown("### 🎯 Prédire le cluster d'un nouveau vin")
        
        # Interface de saisie
        st.markdown("#### 📝 Caractéristiques du vin")
        
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(features):
            with cols[i % 3]:
                # Valeurs par défaut basées sur les données existantes
                min_val = float(st.session_state.df[feature].min())
                max_val = float(st.session_state.df[feature].max())
                mean_val = float(st.session_state.df[feature].mean())
                
                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=0.1
                )
        
        # Prédiction
        if st.button("🎯 Prédire le cluster", type="primary"):
            # Préparation des données
            input_array = np.array([[input_data[feature] for feature in features]])
            
            # Normalisation si applicable
            if scaler is not None:
                input_array = scaler.transform(input_array)
            
            # Prédiction
            cluster = model.predict(input_array)[0]
            probabilities = model.transform(input_array)
            
            st.success(f"✅ Ce vin appartient au **Cluster {cluster}**")
            
            # Affichage des distances
            st.markdown("#### 📏 Distances aux centroïdes")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            distances = probabilities[0]
            clusters_range = range(len(distances))
            
            bars = ax.bar(clusters_range, distances, color='lightcoral', alpha=0.7)
            bars[cluster].set_color('red')
            
            ax.set_title('Distance aux Centroïdes des Clusters')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Distance')
            ax.set_xticks(clusters_range)
            
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(distances):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Interprétation
            st.markdown("#### 📋 Interprétation")
            st.info(f"""
            Le vin avec les caractéristiques saisies est le plus proche du **cluster {cluster}**.
            
            **Caractéristiques typiques de ce cluster :**
            - Vins avec un profil spécifique basé sur les variables analysées
            - Groupe homogène de {len(st.session_state.clustered_data[st.session_state.clustered_data['cluster'] == cluster])} échantillons
            - Distance au centroïde : {distances[cluster]:.2f}
            """)
        
        # Prédiction par fichier
        st.markdown("---")
        st.markdown("### 📁 Prédiction par lot (fichier CSV)")
        
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                
                # Vérification des colonnes
                missing_cols = [col for col in features if col not in new_data.columns]
                if missing_cols:
                    st.error(f"❌ Colonnes manquantes : {missing_cols}")
                else:
                    # Préparation des données
                    X_new = new_data[features]
                    if scaler is not None:
                        X_new_scaled = scaler.transform(X_new)
                    else:
                        X_new_scaled = X_new.values
                    
                    # Prédictions
                    predictions = model.predict(X_new_scaled)
                    new_data['cluster_predicted'] = predictions
                    
                    st.success(f"✅ Prédictions effectuées pour {len(new_data)} échantillons")
                    st.dataframe(new_data, use_container_width=True)
                    
                    # Téléchargement des résultats
                    csv = new_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger les prédictions",
                        data=csv,
                        file_name="wine_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement du fichier : {e}")
                
    except Exception as e:
        display_error("Erreur lors de la prédiction", str(e))

# Fonction principale
def main():
    """Fonction principale de l'application"""
    try:
        # Configuration du thème
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Navigation
        page = sidebar_navigation()
        
        # Routage vers les pages
        page_functions = {
            'Introduction': page_introduction,
            'Exploration des données': page_exploration,
            'Configuration k-means': page_configuration,
            'Entrainement': page_training,
            'Résultats': page_results,
            'Prédiction': page_prediction
        }
        
        if page in page_functions:
            page_functions[page]()
        else:
            st.error(f"Page non trouvée : {page}")
            
    except Exception as e:
        display_error("Erreur critique dans l'application", 
                     f"Une erreur inattendue s'est produite : {str(e)}")

# Point d'entrée
if __name__ == "__main__":
    main()