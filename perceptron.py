import numpy as np


class Perceptron:
    """Perceptron binaire (classes 0 et 1)."""

    def __init__(self, lr=0.1, n_epochs=20):
        # Hyperparamètres : on les choisit AVANT l'entraînement
        self.lr = lr                # taux d'apprentissage (eta) : taille des corrections
        self.n_epochs = n_epochs    # nombre de passages sur tout le jeu de données

        # Paramètres appris : ils seront modifiés PENDANT l'entraînement
        self.weights = None         # créés dans fit(), car leur taille dépend des données
        self.bias = 0.0

    def predict(self, X):
        """Retourne 0 ou 1 pour chaque ligne de X."""
        # Étape 1 : somme pondérée z = w1*x1 + w2*x2 + ... + b, pour tous les exemples d'un coup
        z = X @ self.weights + self.bias
        # Étape 2 : seuil, on prédit 1 si z >= 0, sinon 0
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        """Apprend les poids et le biais à partir de X (n, p) et y (n,) avec y dans {0, 1}."""
        # On démarre sans opinion : poids et biais à zéro
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.errors_ = []  # nombre d'erreurs à chaque époque, pour suivre la convergence

        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                # Prédiction pour un seul exemple (predict attend un tableau 2D)
                y_pred = self.predict(xi.reshape(1, -1))[0]
                # Règle du perceptron : 0 si la prédiction est juste, +-lr si elle est fausse
                update = self.lr * (yi - y_pred)
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self

    def score(self, X, y):
        """Exactitude : part des prédictions correctes."""
        return float(np.mean(self.predict(X) == y))
