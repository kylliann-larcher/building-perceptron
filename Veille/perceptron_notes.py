import random


class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        # poids et biais aléatoires au départ
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = random.uniform(-1, 1)
        self.eta = learning_rate  # taux d'apprentissage

    def step(self, weighted_sum):
        return 1 if weighted_sum > 0 else 0

    def predict(self, x_input):
        weighted_sum = self.bias
        for x, w in zip(x_input, self.weights):
            weighted_sum += x * w
        return self.step(weighted_sum)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):  # on répète sur tous les exemples
            for x_input, target in zip(X, y):
                error = target - self.predict(x_input)  # y - ŷ

                # mise à jour des poids et du biais
                for i in range(len(self.weights)):
                    self.weights[i] += self.eta * error * x_input[i]
                self.bias += self.eta * error


# Test sur des données aléatoires
random.seed(0)

X = [[random.random(), random.random()] for _ in range(100)]
y = [1 if x1 + x2 > 1 else 0 for x1, x2 in X]  # bonne réponse

p = Perceptron(n_inputs=2)
p.train(X, y, epochs=20)

correct = sum(p.predict(x) == t for x, t in zip(X, y))
print("Précision :", correct / len(X))