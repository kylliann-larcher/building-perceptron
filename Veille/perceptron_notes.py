import numpy as np


class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        for epoch in range(self.epochs):
            erreurs = 0
            for xi, yi in zip(X, y):
                erreur = yi - self.predict(xi)  # -1, 0 ou +1
                if erreur != 0:
                    self.w += self.lr * erreur * xi
                    self.b += self.lr * erreur
                    erreurs += 1
            if erreurs == 0:  # convergence
                break
        return self

    def predict(self, X):
        return np.where(X @ self.w + self.b >= 0, 1, 0)


# Données factices : (note maths, note info) -> admis (1) / refusé (0)
rng = np.random.default_rng(0)
X = rng.uniform(0, 20, (100, 2))
y = (X.sum(axis=1) >= 20).astype(int)

X_norm = X / 20  # notes ramenées dans [0, 1]
model = Perceptron(lr=0.1)
model.fit(X_norm, y)

print(model.predict(np.array([[14, 11], [7, 9.5]]) / 20))
print("Exactitude :", np.mean(model.predict(X_norm) == y))
