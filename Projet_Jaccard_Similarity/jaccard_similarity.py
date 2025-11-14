import re # Pour le netoyage des ponctuations des phrases
from typing import Set  # Pour retourner un ensemble de chaîne de caractères

def preprocess_phrase(phrase: str) -> Set[str]:
    """
    Nettoie et tokenise une phrase en un ensemble de mots normalisés.

    Args:
        phrase (str): La phrase à prétraiter.

    Returns:
        Set[str]: Ensemble de mots normalisés (minuscules, sans ponctuation).
    """
    
    # Supprimer la ponctuation et convertir en minuscules
    phrase = re.sub(r'[^\w\s]', '', phrase.lower())
    # Tokeniser en mots et supprimer les mots vides
    mots = phrase.split()
    return set(mots)

def similarite_jaccard(phrase1: str, phrase2: str) -> float:
    """
    Calcule la similarité de Jaccard entre deux phrases.

    Args:
        phrase1 (str): Première phrase.
        phrase2 (str): Deuxième phrase.

    Returns:
        float: Similarité de Jaccard (entre 0 et 1).
    """
    
    # Prétraitement des phrases
    ensemble1 = preprocess_phrase(phrase1)
    ensemble2 = preprocess_phrase(phrase2)

    # Calcul de l'intersection et de l'union
    intersection = ensemble1.intersection(ensemble2)
    union = ensemble1.union(ensemble2)

    # Éviter la division par zéro
    # Si les deux phrases sont vides, retourne 0.0
    if not union:
        return 0.0

    # Similarité de Jaccard
    return len(intersection) / len(union)

def main ():
    print("Calcul de la similarité de jaccard entre 2 phrases :")
    phrase1= input("Saisissez la première phrase : ")
    phrase2= input("Saisissez la deuxième phrase : ")
    similarity= similarite_jaccard(phrase1,phrase2)
    print(f"La similarité de jaccard entres les 2 phrases est : {similarity:.2f}")

if __name__ == "__main__":
    main()