# building-perceptron

Construction d'un perceptron en Python orienté objet, puis application au diagnostic du cancer du sein sur le jeu de données Breast Cancer Wisconsin (Diagnostic).

Projet réalisé en B3 à La Plateforme par Kylliann, Pierre et Yanis.

**Langues :** Français · [English](#english-version)

## Contexte

Le perceptron, proposé par Frank Rosenblatt en 1958, est le premier neurone artificiel. Il calcule une somme pondérée de ses entrées, ajoute un biais, puis applique une fonction seuil pour répondre 0 ou 1. C'est la brique de base des réseaux de neurones actuels.

Le projet comporte trois parties :

1. **Veille** : définir et comparer le Machine Learning et le Deep Learning, présenter trois applications du Deep Learning, puis répondre à sept questions sur le perceptron (définition, équation, règle d'apprentissage, activation, entraînement, limites). Voir [Veille.md](Veille.md).
2. **Implémentation** : écrire une classe `Perceptron` et la tester sur des données aléatoires. Voir [perceptron.py](perceptron.py) et [perceptron.ipynb](perceptron.ipynb).
3. **Application** : prédire si une tumeur est bénigne ou maligne à partir de mesures de noyaux cellulaires, puis juger l'efficacité du perceptron sur ce problème. Voir [analyse_exploratoire.ipynb](analyse_exploratoire.ipynb).

## Structure du dépôt

```text
building-perceptron/
├── perceptron.py               # Classe Perceptron
├── perceptron.ipynb            # Tests sur données aléatoires séparables et sur XOR
├── analyse_exploratoire.ipynb  # Breast Cancer : nettoyage, EDA, réduction, modélisation, conclusion
├── Veille.md                   # Réponses aux questions de veille
├── data/raw/
│   ├── bcw_data.csv            # Données brutes
│   └── bcw_description.txt     # Descriptif des variables
└── requirements.txt
```

## La classe Perceptron

`Perceptron(lr=0.1, n_epochs=20)` est un classifieur binaire (classes 0 et 1) :

- `fit(X, y)` démarre avec des poids et un biais à zéro. Pour chaque exemple, il applique la règle du perceptron : `w ← w + lr × (y − ŷ) × x` et `b ← b + lr × (y − ŷ)`. Le nombre d'erreurs par époque est gardé dans `errors_`.
- `predict(X)` calcule `z = X·w + b` et renvoie 1 si `z ≥ 0`, sinon 0.
- `score(X, y)` renvoie l'exactitude.

Sur des points aléatoires séparables par une droite, les erreurs tombent à 0 en quelques époques. Sur le problème XOR, elles ne tombent jamais à 0 et l'exactitude reste à 0,5. Cela illustre la limite principale du perceptron : il ne trace qu'une frontière linéaire.

## Les données

Le jeu Breast Cancer Wisconsin (Diagnostic) contient **569 échantillons** issus d'images de ponctions à l'aiguille fine : **357 bénins (B)** et **212 malins (M)**. Pour chaque noyau cellulaire, dix caractéristiques sont mesurées : rayon, texture, périmètre, surface, lissage, compacité, concavité, points concaves, symétrie et dimension fractale. Chacune est donnée sous trois formes (moyenne `_mean`, erreur type `_se`, pire valeur `_worst`), soit **30 variables numériques**.

## Démarche et analyse

Tout est dans [analyse_exploratoire.ipynb](analyse_exploratoire.ipynb).

1. **Nettoyage** : suppression de la colonne vide `Unnamed: 32` et de l'identifiant. Il n'y a ni valeur manquante ni doublon. La cible est encodée en `B = 0` et `M = 1`.
2. **Séparation** : 60 % entraînement, 20 % validation, 20 % test, de façon stratifiée (graine 42). Toutes les analyses et tous les ajustements se font sur l'entraînement seul, pour ne pas fausser l'évaluation.
3. **Analyse exploratoire** :
   - Classes déséquilibrées : prédire toujours « bénin » donne déjà environ 63 % d'exactitude. L'exactitude seule ne suffit donc pas.
   - Les mesures de taille et de concavité sont nettement plus élevées chez les tumeurs malignes. La texture et la dimension fractale séparent moins bien les deux classes.
   - Des valeurs extrêmes (méthode IQR) existent surtout chez les cas malins. Ce sont des mesures réelles, donc on les garde.
   - Rayon, périmètre et surface sont presque parfaitement corrélés (r > 0,95) : beaucoup d'information est redondante.
4. **Standardisation** avec la moyenne et l'écart-type de l'entraînement, car le perceptron est sensible aux échelles.
5. **Réduction de dimension**, deux méthodes comparées :
   - **Boruta** (forêt aléatoire) confirme 24 variables. Un filtre de corrélation à 0,90 en garde ensuite **15**, toujours interprétables.
   - **PCA** : **10 composantes** conservent 95,4 % de la variance.
6. **Modélisation** : le perceptron est entraîné sur les 30 variables, sur la sélection Boruta et sur la PCA. Le nombre d'époques et le taux d'apprentissage sont choisis sur la validation, en priorité selon le **rappel**.

## Résultats

Évaluation unique sur le jeu de test (115 échantillons, dont 43 malins). La classe positive est M.

| Modèle | Entrées | Exactitude | Précision | Rappel | Spécificité | F1 | Cancers manqués |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Perceptron, 30 variables | 30 | 98,3 % | 97,7 % | 97,7 % | 98,6 % | 0,977 | 1 / 43 |
| Perceptron, Boruta | 15 | 96,5 % | 97,6 % | 93,0 % | 98,6 % | 0,952 | 3 / 43 |
| Perceptron, PCA | 10 | 98,3 % | 100 % | 95,3 % | 100 % | 0,976 | 2 / 43 |
| Référence : classe majoritaire | 0 | 62,6 % | 0 % | 0 % | 100 % | 0 | 43 / 43 |

Le **rappel** (sensibilité) est la métrique principale : un faux négatif est un cancer annoncé comme bénin, l'erreur la plus grave dans ce contexte. La précision et la spécificité mesurent le coût des fausses alertes.

## Outils utilisés

- **Python 3.12**
- **NumPy** : implémentation du perceptron et calculs
- **pandas** : chargement et manipulation des données
- **Matplotlib** et **seaborn** : visualisations
- **scikit-learn** : PCA, forêt aléatoire, métriques
- **BorutaPy** : sélection de variables
- **Jupyter** : notebooks

## Installation

Utilisez un Python 3.12 installé normalement (python.org ou Anaconda). Sous Windows, évitez la version du Microsoft Store.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

## Conclusion

Un simple perceptron obtient de très bons résultats sur ce jeu de données : jusqu'à 98,3 % d'exactitude et 97,7 % de rappel, contre 0 % de rappel pour la référence. Les deux classes sont donc presque séparables par une frontière linéaire.

Ces résultats sont pourtant à nuancer :

- **Chaque cancer manqué compte.** Même un seul faux négatif sur 43 reste une erreur grave en médecine.
- **La réduction de dimension n'a pas amélioré le modèle.** Avec 10 composantes, la PCA garde une performance proche en divisant le nombre d'entrées par trois. Boruta, préféré sur la validation pour un seul cas d'écart, fait moins bien sur le test.
- **Les jeux sont petits.** Avec 113 et 115 échantillons en validation et en test, un ou deux cas suffisent à changer le classement. Les écarts entre modèles ne sont pas significatifs.
- **Le perceptron a ses limites.** Il ne donne pas de probabilité, ne converge pas si les données ne sont pas parfaitement séparables, et dépend de l'ordre des exemples.

Pistes d'amélioration, non implémentées :

- une régression logistique, pour obtenir une probabilité et ajuster le seuil afin de favoriser le rappel ;
- un SVM, linéaire ou à noyau, pour une frontière à marge maximale ou non linéaire ;
- un perceptron multicouche (MLP), pour apprendre des relations non linéaires ;
- une validation croisée stratifiée, pour mesurer la stabilité des résultats ;
- une pondération des classes, pour pénaliser davantage les faux négatifs ;
- un mélange des exemples à chaque époque et un arrêt anticipé dans la classe `Perceptron`.

## Bibliographie

### Perceptron et apprentissage

- Rosenblatt, F. (1958). *The perceptron: a probabilistic model for information storage and organization in the brain.* Psychological Review, 65(6), 386-408.
- [Cornell Virtual Workshop : The Basic Building Block, The Perceptron](https://cvw.cac.cornell.edu/SciML/mlp/perceptron)
- [Stanford : Neural Networks, Neuron](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Neuron/index.html)
- [Chris Pollett, San José State University : Perceptron Learning](https://www.cs.sjsu.edu/~pollett/256.2.21f/Lec20210908.html)
- [University of Wisconsin-Madison : CS 540 Lecture Notes, Neural Networks](https://pages.cs.wisc.edu/~dyer/cs540/notes/nn.html)
- [Cornell : Introduction to Neural Networks](https://www.cs.cornell.edu/courses/cs4700/2017fa/lectures/pdf/CS4700-15-ML_Neural_Nets_intro_cpg_v6.pdf)

### Machine Learning et Deep Learning

- [Google for Developers : What is Machine Learning?](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml)
- [IBM : What is deep learning?](https://www.ibm.com/think/topics/deep-learning)
- [IBM : AI vs. machine learning vs. deep learning vs. neural networks](https://www.ibm.com/think/topics/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks)
- [Google for Developers : Introduction to decision forests](https://developers.google.com/machine-learning/decision-forests)
- [Keras : Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/)

### Applications du Deep Learning

- [Google : Quick, Draw!](https://quickdraw.withgoogle.com/)
- [Google Magenta : Music Transformer](https://magenta.tensorflow.org/music-transformer)
- [Google DeepMind : AlphaFold](https://deepmind.google/science/alphafold/)
- Jumper, J. et al. (2021). [*Highly accurate protein structure prediction with AlphaFold.*](https://www.nature.com/articles/s41586-021-03819-2) Nature, 596, 583-589.

### Données et méthodes

- Wolberg, W., Mangasarian, O., Street, N. et Street, W. (1993). [*Breast Cancer Wisconsin (Diagnostic).*](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) UCI Machine Learning Repository.
- Street, W. N., Wolberg, W. H. et Mangasarian, O. L. (1993). *Nuclear feature extraction for breast tumor diagnosis.* IS&T/SPIE Symposium on Electronic Imaging, vol. 1905, 861-870.
- Kursa, M. B. et Rudnicki, W. R. (2010). *Feature Selection with the Boruta Package.* Journal of Statistical Software, 36(11), 1-13.
- [BorutaPy](https://github.com/scikit-learn-contrib/boruta_py)
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.

---

## English version

Building a perceptron in object-oriented Python, then applying it to breast cancer diagnosis with the Breast Cancer Wisconsin (Diagnostic) dataset.

Project carried out in the third year (B3) at La Plateforme by Kylliann, Pierre and Yanis.

### Context

The perceptron, introduced by Frank Rosenblatt in 1958, is the first artificial neuron. It computes a weighted sum of its inputs, adds a bias, then applies a step function to output 0 or 1. It is the basic building block of today's neural networks.

The project has three parts:

1. **Research**: define and compare Machine Learning and Deep Learning, present three Deep Learning applications, then answer seven questions about the perceptron (definition, equation, learning rule, activation, training, limitations). See [Veille.md](Veille.md) (in French).
2. **Implementation**: write a `Perceptron` class and test it on random data. See [perceptron.py](perceptron.py) and [perceptron.ipynb](perceptron.ipynb).
3. **Application**: predict whether a tumour is benign or malignant from cell nucleus measurements, then assess how well the perceptron handles this problem. See [analyse_exploratoire.ipynb](analyse_exploratoire.ipynb).

The notebooks and code comments are written in French.

### Repository structure

```text
building-perceptron/
├── perceptron.py               # Perceptron class
├── perceptron.ipynb            # Tests on linearly separable random data and on XOR
├── analyse_exploratoire.ipynb  # Breast Cancer: cleaning, EDA, dimensionality reduction, modelling, conclusion
├── Veille.md                   # Answers to the research questions
├── data/raw/
│   ├── bcw_data.csv            # Raw data
│   └── bcw_description.txt     # Variable description
└── requirements.txt
```

### The Perceptron class

`Perceptron(lr=0.1, n_epochs=20)` is a binary classifier (classes 0 and 1):

- `fit(X, y)` starts with zero weights and bias. For each sample, it applies the perceptron rule: `w ← w + lr × (y − ŷ) × x` and `b ← b + lr × (y − ŷ)`. The number of errors per epoch is stored in `errors_`.
- `predict(X)` computes `z = X·w + b` and returns 1 if `z ≥ 0`, otherwise 0.
- `score(X, y)` returns the accuracy.

On random points that a straight line can separate, errors drop to 0 within a few epochs. On the XOR problem, they never reach 0 and accuracy stays at 0.5. This shows the perceptron's main limitation: it can only draw a linear boundary.

### Data

The Breast Cancer Wisconsin (Diagnostic) dataset contains **569 samples** from fine needle aspirate images: **357 benign (B)** and **212 malignant (M)**. Ten features are measured for each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each one is given in three forms (mean `_mean`, standard error `_se`, worst value `_worst`), for a total of **30 numerical variables**.

### Approach and analysis

Everything is in [analyse_exploratoire.ipynb](analyse_exploratoire.ipynb).

1. **Cleaning**: the empty column `Unnamed: 32` and the ID are removed. There are no missing values or duplicates. The target is encoded as `B = 0` and `M = 1`.
2. **Split**: 60% training, 20% validation, 20% test, stratified (seed 42). All analysis and fitting use the training set only, so the evaluation stays unbiased.
3. **Exploratory analysis**:
   - Imbalanced classes: always predicting "benign" already gives about 63% accuracy, so accuracy alone is not enough.
   - Size and concavity measurements are clearly higher for malignant tumours. Texture and fractal dimension separate the two classes less well.
   - Outliers (IQR method) appear mostly in malignant cases. They are real measurements, so they are kept.
   - Radius, perimeter and area are almost perfectly correlated (r > 0.95): much of the information is redundant.
4. **Standardisation** with the training mean and standard deviation, because the perceptron is sensitive to feature scales.
5. **Dimensionality reduction**, two methods compared:
   - **Boruta** (random forest) confirms 24 variables. A correlation filter at 0.90 then keeps **15**, which remain interpretable.
   - **PCA**: **10 components** retain 95.4% of the variance.
6. **Modelling**: the perceptron is trained on the 30 variables, on the Boruta selection and on the PCA components. The number of epochs and the learning rate are chosen on the validation set, with **recall** as the main criterion.

### Results

Single evaluation on the test set (115 samples, including 43 malignant). The positive class is M.

| Model | Inputs | Accuracy | Precision | Recall | Specificity | F1 | Missed cancers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Perceptron, 30 variables | 30 | 98.3% | 97.7% | 97.7% | 98.6% | 0.977 | 1 / 43 |
| Perceptron, Boruta | 15 | 96.5% | 97.6% | 93.0% | 98.6% | 0.952 | 3 / 43 |
| Perceptron, PCA | 10 | 98.3% | 100% | 95.3% | 100% | 0.976 | 2 / 43 |
| Baseline: majority class | 0 | 62.6% | 0% | 0% | 100% | 0 | 43 / 43 |

**Recall** (sensitivity) is the main metric: a false negative is a cancer reported as benign, the most serious error in this context. Precision and specificity measure the cost of false alarms.

### Tools

- **Python 3.12**
- **NumPy**: perceptron implementation and computations
- **pandas**: data loading and manipulation
- **Matplotlib** and **seaborn**: visualisations
- **scikit-learn**: PCA, random forest, metrics
- **BorutaPy**: feature selection
- **Jupyter**: notebooks

### Installation

Use a regular Python 3.12 installation (python.org or Anaconda). On Windows, avoid the Microsoft Store version.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

### Conclusion

A simple perceptron performs very well on this dataset: up to 98.3% accuracy and 97.7% recall, compared with 0% recall for the baseline. The two classes are therefore almost linearly separable.

These results still need to be put in perspective:

- **Every missed cancer matters.** Even one false negative out of 43 is a serious error in medicine.
- **Dimensionality reduction did not improve the model.** With 10 components, PCA keeps similar performance with three times fewer inputs. Boruta, preferred on the validation set by a single case, does worse on the test set.
- **The datasets are small.** With 113 and 115 samples for validation and test, one or two cases are enough to change the ranking. The differences between models are not significant.
- **The perceptron has limits.** It gives no probability, does not converge if the data are not perfectly separable, and depends on the order of the samples.

Possible improvements, not implemented:

- logistic regression, to get a probability and tune the threshold in favour of recall;
- an SVM, linear or kernel-based, for a maximum-margin or non-linear boundary;
- a multilayer perceptron (MLP), to learn non-linear relationships;
- stratified cross-validation, to measure how stable the results are;
- class weighting, to penalise false negatives more heavily;
- shuffling the samples at each epoch and early stopping in the `Perceptron` class.

### Bibliography

See the [Bibliographie](#bibliographie) section above. The references are the same in both languages.
