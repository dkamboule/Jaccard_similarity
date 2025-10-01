# Projet : Impléménter en Python la similarité de Jaccard pour les Phrases

## Description
Ce projet implémente la **similarité de Jaccard** pour comparer des phrases en Python. La similarité de Jaccard est une métrique utilisée pour mesurer la similarité entre deux ensembles. Ici, elle est appliquée aux phrases en les convertissant en ensembles de mots.
**Gestion de l'élision**: dans cette implémentation de la similarité de Jaccard, un mot élidé est considéré comme un seul mot et non comme deux mots. Exemple : l'étudiant est considéré comme un seul mot (l'étudiant) et non comme deux mots; l' et étudiant.
Il est tout à fait possible de traiter l'élision autrement, mais cela ne fait pas l'objet de ce projet.

## Fonctionnalités
- Prétraitement des phrases (suppression de la ponctuation, tokenisation, normalisation).
- Calcul de la similarité de Jaccard entre deux phrases.
- Gestion des cas limites (phrases vides, ensembles vides).
- Interaction avec l'utilisateur pour la saisie des phrases.

## Prérequis
- Python 3.6 ou supérieur.

## Installation
1. Clonez ce dépôt ou téléchargez les fichiers.
2. Assurez-vous d'avoir Python installé.

## Exécution
Pour exécuter le programme principal sur le terminal, se placer dans le repertoire du fichier puis exécuter : python jaccard_similarity.py

## Test unitaire
Le projet contient un fichier de test unitaire pour  vérifier que la fonction de calcul  de la similarité de Jaccard fonctionne correctement dans divers scénarios.

Pour exécuter le fichier du test unitaire sur le terminal, se placer dans le repertoire du fichier puis exécuter : python test_jaccard_similarity.py


## Exemple d'utilisation

```python
from jaccard_similarity import similarite_jaccard

phrase1 = "Le machine learning supervisé traite des jeux de données étiquetées."
phrase2 = "Le machine learning non supervisé traite des jeux de données non étiquetées."
similarite = similarite_jaccard(phrase1, phrase2)
print(f"Similarité : {similarite:.2f}")