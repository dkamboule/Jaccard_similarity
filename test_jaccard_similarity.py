import unittest
from jaccard_similarity import similarite_jaccard

class TestSimilariteJaccard(unittest.TestCase):
    """Tests unitaires pour la fonction de similarité de Jaccard."""

    def test_phrases_identiques(self):
        """Test avec deux phrases identiques."""
        phrase1 = "Bonjour le monde."
        phrase2 = "Bonjour le monde."
        self.assertAlmostEqual(similarite_jaccard(phrase1, phrase2), 1.0)

    def test_phrases_partiellement_similaires(self):
        """Test avec deux phrases partiellement similaires."""
        phrase1 = "Le chat dort sur le tapis."
        phrase2 = "Le chien dort sur le tapis."
        self.assertAlmostEqual(similarite_jaccard(phrase1, phrase2), 2/3, places=2)

    def test_phrases_sans_mots_communs(self):
        """Test avec deux phrases sans mots communs."""
        phrase1 = "Je mange une pomme."
        phrase2 = "Tu bois de l'eau."
        self.assertAlmostEqual(similarite_jaccard(phrase1, phrase2), 0.0)

    def test_phrases_vides(self):
        """Test avec deux phrases vides."""
        phrase1 = ""
        phrase2 = ""
        self.assertAlmostEqual(similarite_jaccard(phrase1, phrase2), 0.0)

    def test_phrase_vide_et_non_vide(self):
        """Test avec une phrase vide et une phrase non vide."""
        phrase1 = ""
        phrase2 = "Bonjour le monde."
        self.assertAlmostEqual(similarite_jaccard(phrase1, phrase2), 0.0)

if __name__ == "__main__":
    unittest.main()
