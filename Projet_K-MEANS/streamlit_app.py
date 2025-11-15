"""
Application Streamlit pour l'analyse K-means du dataset Wine Quality
"""

import streamlit as st # Interface utilisateur
import pandas as pd # Manipulation des données
import numpy as np # Calculs numériques
import matplotlib.pyplot as plt # Visualisation
import seaborn as sns # Visualisation avancée
from sklearn.preprocessing import StandardScaler # Normalisation des données
from sklearn.cluster import KMeans # Algorithme K-means
from sklearn.decomposition import PCA # Réduction de dimension
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # Métriques d'évaluation
from sklearn.model_selection import KFold # Validation croisée
import joblib # Sauvegarde et chargement de modèles
import pickle # Sérialisation d'objets
import plotly.express as px # Visualisation interactive
import plotly.graph_objects as go # Graphiques interactifs
from plotly.subplots import make_subplots # Sous-graphiques interactifs
import warnings # Gestion des avertissements
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
</style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'page' not in st.session_state:
    st.session_state.page = 'Introduction'
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None

# Fonction de chargement des données
@st.cache_data
def load_data():
    """Charge et nettoie le dataset"""
    try:
        df = pd.read_csv('winequality-white.csv', sep=';',decimal='.')
        df = df.drop_duplicates()
        return df
    except FileNotFoundError:
        st.error("❌ Le fichier 'winequality-white.csv' n'a pas été trouvé.")
        return None

# Fonction de validation croisée
def cross_validate_kmeans(X, k_range, n_splits=5):
    """Validation croisée pour sélectionner le meilleur k"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {k: {'silhouette': [], 'davies_bouldin': [], 'calinski': []} for k in k_range}
    
    progress_bar = st.progress(0)
    total_iterations = len(k_range) * n_splits
    current_iteration = 0
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_train)
            val_labels = kmeans.predict(X_val)
            
            if len(np.unique(val_labels)) > 1:
                results[k]['silhouette'].append(silhouette_score(X_val, val_labels))
                results[k]['davies_bouldin'].append(davies_bouldin_score(X_val, val_labels))
                results[k]['calinski'].append(calinski_harabasz_score(X_val, val_labels))
            
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
    
    progress_bar.empty()
    
    # Calcul des moyennes
    cv_results = {}
    for k in k_range:
        cv_results[k] = {
            'silhouette_mean': np.mean(results[k]['silhouette']),
            'silhouette_std': np.std(results[k]['silhouette']),
            'davies_bouldin_mean': np.mean(results[k]['davies_bouldin']),
            'calinski_mean': np.mean(results[k]['calinski'])
        }
    
    return cv_results

# Menu de navigation
def sidebar_navigation():
    """Menu de navigation dans la sidebar"""
    st.sidebar.title("🍷 Navigation")
    pages = ['Introduction', 'Exploration des données', 'Configuration k-means', 
             'Entrainement', 'Résultats', 'Prédiction']
    
    selected_page = st.sidebar.radio("Choisir une page", pages)
    st.session_state.page = selected_page
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 État du modèle")
    if st.session_state.model_trained:
        st.sidebar.success("✅ Modèle entrainé")
    else:
        st.sidebar.info("⏳ Modèle non entrainé")
    
    return selected_page

# Page d'introduction
def page_introduction():
    """Page d'introduction du projet"""
    st.markdown('<h1 class="main-header">🍷 Analyse k-means du dataset wine quality</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=800", 
                caption="Analyse de la qualité du vin par clustering", width='stretch')
    
    st.markdown("---")
    
    # Description du projet
    st.markdown("## 📋 Description du projet")
    st.info("""
    Cette application interactive permet d'analyser le dataset **wine quality** en utilisant 
    l'algorithme de clustering **k-means**. L'objectif est de segmenter les vins en groupes 
    homogènes basés sur leurs caractéristiques physico-chimiques.
    """)
    
    # Objectifs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Objectifs")
        st.markdown("""
        - **Explorer** les caractéristiques physico-chimiques des vins
        - **Identifier** des profils de vins distincts
        - **Analyser** la relation entre clusters et qualité
        - **Optimiser** le nombre de clusters via validation croisée
        - **Prédire** l'appartenance de nouveaux vins aux clusters
        """)
    
    with col2:
        st.markdown("### 📊 Dataset")
        st.markdown("""
        - **Source**: UCI Machine Learning Repository
        - **Type**: Vins blancs portugais (Vinho Verde)
        - **Variables**: 11 caractéristiques physico-chimiques
        - **Cible**: Qualité (score de 0 à 10)
        - **Taille**: 4898 échantillons
        """)
    
    # Méthodologie
    st.markdown("---")
    st.markdown("## 🔬 Méthodologie")
    
    method_tabs = st.tabs(["1️⃣ Prétraitement", "2️⃣ Clustering", "3️⃣ Evaluation"])
    
    with method_tabs[0]:
        st.markdown("""
        #### Etapes de prétraitement:
        1. **Nettoyage des données**: Suppression des doublons et valeurs manquantes
        2. **Détection d'outliers**: Méthode IQR (Interquartile Range)
        3. **Normalisation**: StandardScaler pour uniformiser les échelles
        4. **Réduction de dimension**: PCA pour visualisation
        """)
    
    with method_tabs[1]:
        st.markdown("""
        #### Configuration du k-means:
        1. **Sélection de k**: Méthode du coude et score de silhouette
        2. **Validation croisée**: 5-fold pour robustesse
        3. **Optimisation**: Multiple initialisations (n_init)
        4. **Convergence**: Critères d'arrêt adaptés
        """)
    
    with method_tabs[2]:
        st.markdown("""
        #### Métriques d'évaluation:
        - **Score de silhouette**: Mesure de cohésion et séparation
        - **Davies-Bouldin**: Ratio de dispersion intra/inter-cluster
        - **Calinski-Harabasz**: Ratio de variance inter/intra-cluster
        - **Analyse qualitative**: Interprétation des profils
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Développé dans le cadre du cours de ML non supervisé - Master IFOAD/UJKZ</p>
    </div>
    """, unsafe_allow_html=True)

# Page d'exploration des données
def page_exploration():
    """Page d'exploration des données"""
    st.markdown('<h1 class="main-header">📊 Exploration des données</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    # Informations générales
    st.markdown("## 📋 Informations générales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de lignes", f"{len(df):,}")
    with col2:
        st.metric("Nombre de colonnes", df.shape[1])
    with col3:
        st.metric("Valeurs manquantes", df.isnull().sum().sum())
    with col4:
        st.metric("Doublons", df.duplicated().sum())
    
    # Aperçu des données
    with st.expander("👀 Aperçu des données", expanded=True):
        st.dataframe(df.head(10), width='stretch')
    
    # Statistiques descriptives
    st.markdown("## 📈 Statistiques descriptives")
    st.dataframe(df.describe(), width='stretch')
    
    # Distributions
    st.markdown("## 📊 Distributions des variables")
    
    feature_names = df.columns[:-1].tolist()
    selected_features = st.multiselect(
        "Sélectionner les variables à visualiser",
        feature_names,
        default=feature_names[:4]
    )
    
    if selected_features:
        fig = make_subplots(
            rows=(len(selected_features) + 1) // 2,
            cols=2,
            subplot_titles=selected_features
        )
        
        for idx, col in enumerate(selected_features):
            row = idx // 2 + 1
            col_idx = idx % 2 + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=30, showlegend=False),
                row=row, col=col_idx
            )
        
        fig.update_layout(height=300 * ((len(selected_features) + 1) // 2), showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    # Matrice de corrélation
    st.markdown("## 🔥 Matrice de corrélation")
    
    corr_matrix = df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, width='stretch')
    
    # Corrélations avec la qualité
    st.markdown("### 🎯 Corrélations avec la qualité")
    quality_corr = corr_matrix['quality'].sort_values(ascending=False)[1:]
    
    fig_bar = px.bar(
        x=quality_corr.values,
        y=quality_corr.index,
        orientation='h',
        color=quality_corr.values,
        color_continuous_scale='RdBu_r',
        labels={'x': 'Corrélation', 'y': 'Variable'}
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, width='stretch')

# Page de configuration k-means
def page_configuration():
    """Page de configuration des paramètres k-means"""
    st.markdown('<h1 class="main-header">⚙️ Configuration k-means</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    st.markdown("## 🎛️ Paramètres du modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Paramètres de base")
        k_min = st.slider("k minimum", 2, 5, 2)
        k_max = st.slider("k maximum", 6, 15, 10)
        n_init = st.slider("Nombre d'initialisations", 5, 50, 10)
        max_iter = st.slider("Nombre max d'itérations", 100, 500, 300)
    
    with col2:
        st.markdown("### Options avancées")
        algorithm = st.selectbox("Algorithme", ['lloyd', 'elkan', 'auto'])
        random_state = st.number_input("Random state", 0, 100, 42)
        tol = st.select_slider("Tolérance", [1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
        remove_outliers = st.checkbox("Retirer les outliers (IQR > 3.0)", value=False)
    
    st.markdown("---")
    st.markdown("## 🔍 Sélection du nombre optimal de clusters")
    
    if st.button("🚀 Lancer l'analyse pour sélection de k", type="primary"):
        with st.spinner("Analyse en cours..."):
            # Préparation des données
            X = df.drop('quality', axis=1)
            
            if remove_outliers:
                Q1 = X.quantile(0.25)
                Q3 = X.quantile(0.75)
                IQR = Q3 - Q1
                outliers_mask = ((X < Q1 - 3.0 * IQR) | (X > Q3 + 3.0 * IQR)).any(axis=1)
                X = X[~outliers_mask]
                st.info(f"📌 {outliers_mask.sum()} outliers retirés")
            
            # Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calcul des métriques
            k_range = range(k_min, k_max + 1)
            metrics = {
                'k': list(k_range),
                'inertia': [],
                'silhouette': [],
                'davies_bouldin': [],
                'calinski': []
            }
            
            progress = st.progress(0)
            for i, k in enumerate(k_range):
                kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, 
                              algorithm=algorithm, random_state=random_state, tol=tol)
                labels = kmeans.fit_predict(X_scaled)
                
                metrics['inertia'].append(kmeans.inertia_)
                metrics['silhouette'].append(silhouette_score(X_scaled, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
                metrics['calinski'].append(calinski_harabasz_score(X_scaled, labels))
                
                progress.progress((i + 1) / len(k_range))
            
            progress.empty()
            
            # Visualisation des résultats
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Méthode du coude', 'Score de silhouette',
                              'Score Davies-Bouldin', 'Score Calinski-Harabasz')
            )
            
            fig.add_trace(go.Scatter(x=metrics['k'], y=metrics['inertia'], mode='lines+markers',
                                    name='Inertie'), row=1, col=1)
            fig.add_trace(go.Scatter(x=metrics['k'], y=metrics['silhouette'], mode='lines+markers',
                                    name='Silhouette'), row=1, col=2)
            fig.add_trace(go.Scatter(x=metrics['k'], y=metrics['davies_bouldin'], mode='lines+markers',
                                    name='Davies-Bouldin'), row=2, col=1)
            fig.add_trace(go.Scatter(x=metrics['k'], y=metrics['calinski'], mode='lines+markers',
                                    name='Calinski'), row=2, col=2)
            
            fig.update_xaxes(title_text="Nombre de clusters (k)")
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, width='stretch')
            
            # Recommandation
            optimal_k = metrics['k'][np.argmax(metrics['silhouette'])]
            st.success(f"✨ k optimal recommandé: **{optimal_k}** (basé sur le score de silhouette)")
            
            # Sauvegarde des paramètres
            st.session_state.config = {
                'k': optimal_k,
                'n_init': n_init,
                'max_iter': max_iter,
                'algorithm': algorithm,
                'random_state': random_state,
                'tol': tol,
                'remove_outliers': remove_outliers
            }

# Page d'entraînement
def page_training():
    """Page d'entraînement du modèle"""
    st.markdown('<h1 class="main-header">🎓 Entrainement du modèle</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        return
    
    if 'config' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord configurer les paramètres dans la page configuration!")
        return
    
    config = st.session_state.config
    
    # Affichage des paramètres
    st.markdown("## 📋 Paramètres sélectionnés")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre de clusters (k)", config['k'])
        st.metric("Algorithme", config['algorithm'])
    with col2:
        st.metric("Initialisations", config['n_init'])
        st.metric("Max itérations", config['max_iter'])
    with col3:
        st.metric("Random state", config['random_state'])
        st.metric("Outliers retirés", "Oui" if config['remove_outliers'] else "Non")
    
    st.markdown("---")
    
    # Options de validation croisée
    st.markdown("## 🔄 Validation croisée")
    use_cv = st.checkbox("Utiliser la validation croisée pour optimiser K", value=True)
    
    if use_cv:
        col1, col2 = st.columns(2)
        with col1:
            cv_folds = st.slider("Nombre de folds", 3, 10, 5)
        with col2:
            k_range_cv = st.slider("Range de k pour CV", 2, 15, (config['k']-1, config['k']+12))
    
    # Bouton d'entraînement
    if st.button("🚀 Lancer l'entrainement", type="primary", width='stretch'):
        with st.spinner("Entrainement en cours..."):
            
            # Préparation des données
            X = df.drop('quality', axis=1)
            y = df['quality']
            
            if config['remove_outliers']:
                Q1 = X.quantile(0.25)
                Q3 = X.quantile(0.75)
                IQR = Q3 - Q1
                outliers_mask = ((X < Q1 - 3.0 * IQR) | (X > Q3 + 3.0 * IQR)).any(axis=1)
                X = X[~outliers_mask]
                y = y[~outliers_mask]
                df_clean = df[~outliers_mask]
            else:
                df_clean = df.copy()
            
            # Standardisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Validation croisée
            if use_cv:
                st.info("🔄 Validation croisée en cours...")
                k_range = range(k_range_cv[0], k_range_cv[1] + 1)
                cv_results = cross_validate_kmeans(X_scaled, k_range, n_splits=cv_folds)
                
                # Affichage des résultats CV
                cv_df = pd.DataFrame(cv_results).T
                best_k = cv_df['silhouette_mean'].idxmax()
                
                st.success(f"✅ Meilleur k selon CV: **{best_k}**")
                
                # Graphique des résultats CV
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(k_range),
                    y=cv_df['silhouette_mean'],
                    mode='lines+markers',
                    name='Score moyen',
                    error_y=dict(
                        type='data',
                        array=cv_df['silhouette_std'],
                        visible=True
                    )
                ))
                fig.update_layout(
                    title="Score de silhouette par validation croisée",
                    xaxis_title="Nombre de clusters (k)",
                    yaxis_title="Score de silhouette",
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
                
                final_k = best_k
            else:
                final_k = config['k']
            
            # Entrainement final
            st.info(f"🎯 Entrainement du modèle final avec k={final_k}...")
            
            kmeans = KMeans(
                n_clusters=final_k,
                n_init=config['n_init'],
                max_iter=config['max_iter'],
                algorithm=config['algorithm'],
                random_state=config['random_state'],
                tol=config['tol']
            )
            
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calcul des métriques
            sil_score = silhouette_score(X_scaled, cluster_labels)
            db_score = davies_bouldin_score(X_scaled, cluster_labels)
            ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
            
            # Affichage des métriques
            st.markdown("### 📊 Métriques de performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Score de silhouette", f"{sil_score:.3f}", 
                         help="Plus élevé = meilleur (range: -1 à 1)")
            with col2:
                st.metric("Score Davies-Bouldin", f"{db_score:.3f}",
                         help="Plus bas = meilleur")
            with col3:
                st.metric("Score Calinski-Harabasz", f"{ch_score:.0f}",
                         help="Plus élevé = meilleur")
            
            # Sauvegarde du modèle
            st.session_state.kmeans_model = kmeans
            st.session_state.scaler = scaler
            st.session_state.model_trained = True
            st.session_state.final_k = final_k
            
            # Ajout des clusters au dataframe
            df_clean['cluster'] = cluster_labels
            st.session_state.clustered_data = df_clean
            st.session_state.X_scaled = X_scaled
            
            # Sauvegarde sur disque
            with st.spinner("Sauvegarde des modèles..."):
                joblib.dump(kmeans, 'kmeans_model.joblib')
                joblib.dump(scaler, 'scaler.joblib')
                
                with open('kmeans_model.pkl', 'wb') as f:
                    pickle.dump(kmeans, f)
                with open('scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                
                metadata = {
                    'optimal_k': final_k,
                    'features': X.columns.tolist(),
                    'silhouette_score': sil_score,
                    'davies_bouldin_score': db_score,
                    'calinski_harabasz_score': ch_score,
                    'n_samples_trained': len(X_scaled),
                    'config': config
                }
                
                with open('model_metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
            
            st.success("✅ Modèle entrainé et sauvegardé avec succès!")
            st.balloons()

# Page de résultats
def page_results():
    """Page d'affichage des résultats"""
    st.markdown("""
    <h1 class="main-header">📊 Résultats de l'analyse</h1>
    """, unsafe_allow_html=True)

    
    if not st.session_state.model_trained:
        st.warning("⚠️ Veuillez d'abord entrainer le modèle")
        return
    
    df_clustered = st.session_state.clustered_data
    X_scaled = st.session_state.X_scaled
    kmeans = st.session_state.kmeans_model
    
    # Statistiques des clusters
    st.markdown("## 📈 Statistiques des clusters")
    
    cluster_stats = df_clustered.groupby('cluster').agg({
        'quality': ['mean', 'std', 'count']
    }).round(2)
    cluster_stats.columns = ['Qualité moyenne', 'Ecart-type', 'Nombre de vins']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(cluster_stats, width='stretch')
    
    with col2:
        # Graphique en camembert
        fig_pie = px.pie(
            values=cluster_stats['Nombre de vins'],
            names=cluster_stats.index,
            title="Répartition des vins par cluster"
        )
        st.plotly_chart(fig_pie, width='stretch')
    
    # Visualisation PCA
    st.markdown("## 🎨 Visualisation des clusters (PCA)")
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig_pca = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=df_clustered['cluster'].astype(str),
        title=f"Projection PCA des clusters (variance expliquée: {pca.explained_variance_ratio_.sum():.1%})",
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Ajout des centres
    centers_pca = pca.transform(kmeans.cluster_centers_)
    for i, center in enumerate(centers_pca):
        fig_pca.add_scatter(
            x=[center[0]], y=[center[1]],
            mode='markers',
            marker=dict(size=20, symbol='star', color='black'),
            name=f'Centre {i}',
            showlegend=False
        )
    
    st.plotly_chart(fig_pca, width='stretch')
    
    # Profils des clusters
    st.markdown("## 🔬 Profils des clusters")
    
    feature_cols = df_clustered.columns[:-2]  # Exclusion de 'quality' et de 'cluster'
    cluster_profiles = df_clustered.groupby('cluster')[feature_cols].mean()
    
    # Normalisation pour visualisation
    cluster_profiles_norm = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
    
    fig_heatmap = px.imshow(
        cluster_profiles_norm.T,
        labels=dict(x="Cluster", y="Variable", color="Valeur normalisée"),
        color_continuous_scale='RdBu_r',
        aspect='auto',
        text_auto='.2f'
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, width='stretch')
    
    # Analyse détaillée par cluster
    st.markdown("## 📝 Analyse détaillée par cluster")
    
    selected_cluster = st.selectbox("Sélectionner un cluster", range(st.session_state.final_k))
    
    cluster_data = df_clustered[df_clustered['cluster'] == selected_cluster]
    overall_mean = df_clustered[feature_cols].mean()
    cluster_mean = cluster_data[feature_cols].mean()
    diff_percent = ((cluster_mean - overall_mean) / overall_mean * 100).sort_values()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### Cluster {selected_cluster}")
        st.metric("Taille", f"{len(cluster_data)} vins ({len(cluster_data)/len(df_clustered)*100:.1f}%)")
        st.metric("Qualité moyenne", f"{cluster_data['quality'].mean():.2f} ± {cluster_data['quality'].std():.2f}")
    
    with col2:
        # Variables distinctives
        st.markdown("### Variables distinctives")
        
        # Top 3 sous la moyenne
        st.markdown("**📉 Sous la moyenne:**")
        for var in diff_percent.head(3).index:
            st.write(f"- {var}: {diff_percent[var]:.1f}%")
        
        # Top 3 au-dessus de la moyenne
        st.markdown("**📈 Au-dessus de la moyenne:**")
        for var in diff_percent.tail(3).index:
            st.write(f"- {var}: {diff_percent[var]:.1f}%")
    
    # Distribution de la qualité par cluster
    st.markdown("## 🍷 Qualité par cluster")
    
    fig_box = px.box(
        df_clustered,
        x='cluster',
        y='quality',
        title="Distribution de la qualité par cluster",
        color='cluster',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width='stretch')

# Page de prédiction
def page_prediction():
    """Page de prédiction pour nouveaux échantillons"""
    st.markdown('<h1 class="main-header">🔮 Prédiction de cluster</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Veuillez d'abord entrainer le modèle")
        return
    
    st.markdown("## 📝 Entrer les caractéristiques du vin")
    
    df = load_data()
    feature_cols = df.columns[:-1].tolist()
    
    # Formulaire d'entrée
    col1, col2, col3 = st.columns(3)
    
    input_values = {}
    for i, feature in enumerate(feature_cols):
        col_idx = i % 3
        if col_idx == 0:
            with col1:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(df[feature].mean()),
                    format="%.4f"
                )
        elif col_idx == 1:
            with col2:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(df[feature].mean()),
                    format="%.4f"
                )
        else:
            with col3:
                input_values[feature] = st.number_input(
                    feature,
                    value=float(df[feature].mean()),
                    format="%.4f"
                )
    
    # Bouton de prédiction
    if st.button("🔮 Prédire le cluster", type="primary", width='stretch'):
        # Préparation des données
        input_df = pd.DataFrame([input_values])
        input_scaled = st.session_state.scaler.transform(input_df)
        
        # Prédiction
        cluster_pred = st.session_state.kmeans_model.predict(input_scaled)[0]
        
        # Affichage du résultat
        st.success(f"🎯 Ce vin appartient au **cluster {cluster_pred}**")
        
        # Caractéristiques du cluster
        cluster_data = st.session_state.clustered_data[
            st.session_state.clustered_data['cluster'] == cluster_pred
        ]
        
        st.markdown("### 📊 Caractéristiques du cluster prédit")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Taille du cluster", f"{len(cluster_data)} vins")
        with col2:
            st.metric("Qualité moyenne", f"{cluster_data['quality'].mean():.2f}")
        with col3:
            st.metric("Écart-type qualité", f"{cluster_data['quality'].std():.2f}")
        
        # Comparaison avec le profil moyen du cluster
        st.markdown("### 📈 Comparaison avec le profil moyen du cluster")
        
        cluster_mean = cluster_data[feature_cols].mean()
        comparison_df = pd.DataFrame({
            'Variable': feature_cols,
            'Valeur entrée': [input_values[f] for f in feature_cols],
            'Moyenne du cluster': cluster_mean.values,
            'Différence (%)': [((input_values[f] - cluster_mean[f]) / cluster_mean[f] * 100) 
                               for f in feature_cols]
        })
        comparison_df = comparison_df.round(3)
        
        st.dataframe(comparison_df, width='stretch')
        
        # Graphique radar
        fig_radar = go.Figure()
        
        # Normalisation des valeurs pour le radar
        input_norm = st.session_state.scaler.transform(input_df)[0]
        cluster_mean_norm = st.session_state.scaler.transform([cluster_mean.values])[0]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=input_norm,
            theta=feature_cols,
            fill='toself',
            name='Vin entré',
            line_color='blue'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_mean_norm,
            theta=feature_cols,
            fill='toself',
            name=f'Moyenne Cluster {cluster_pred}',
            line_color='red',
            opacity=0.6
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-2, 2])
            ),
            showlegend=True,
            title="Profil du vin vs moyenne du cluster (normalisé)"
        )
        
        st.plotly_chart(fig_radar, width='stretch')

# Fonction principale
def main():
    """Fonction principale de l'application"""
    # Configuration du thème
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Navigation
    page = sidebar_navigation()
    
    # Routage vers les pages
    if page == 'Introduction':
        page_introduction()
    elif page == 'Exploration des données':
        page_exploration()
    elif page == 'Configuration k-means':
        page_configuration()
    elif page == 'Entrainement':
        page_training()
    elif page == 'Résultats':
        page_results()
    elif page == 'Prédiction':
        page_prediction()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: gray;'>
        <p>Analyse du dataset wine quality à l'aide de l'algorithme des k-means </p>
        <p>© Groupe 5 - ML non supervisé - Master IFOAD/UJKZ</p>
    </div>
    """, unsafe_allow_html=True)

# Point d'entrée
if __name__ == "__main__":
    main()