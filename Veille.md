# Veille : Machine Learning, Deep Learning et perceptron

# Partie 1 : Machine Learning et Deep Learning

## 1. Définissez les notions de Machine Learning et de Deep Learning puis comparez-les. À quel moment doit-on utiliser l'un plutôt que l'autre ?

### Définitions

Le Machine Learning (ML), ou apprentissage automatique, est une branche de l'intelligence artificielle. Son principe est d'entraîner un modèle à partir de données pour qu'il apprenne des relations et puisse ensuite faire des prédictions sur de nouveaux exemples. On ne lui fournit donc pas toutes les règles à appliquer : il les apprend pendant l'entraînement. Par exemple, un modèle peut apprendre à estimer le prix d'un logement à partir de sa surface, de sa localisation et de son nombre de pièces. [1]

L'apprentissage peut être supervisé, lorsque les exemples sont accompagnés des réponses attendues, ou non supervisé, lorsque le modèle cherche des structures dans des données sans réponse fournie. Il existe aussi l'apprentissage par renforcement, dans lequel un agent apprend à agir grâce aux récompenses reçues lors de ses interactions avec un environnement. [1]

Le Deep Learning (DL), ou apprentissage profond, fait partie du Machine Learning. Il repose sur des réseaux de neurones artificiels composés de plusieurs couches de traitement. Pendant l'entraînement, le modèle ajuste les poids des connexions pour réduire ses erreurs. [2]

L'intérêt de ces couches est d'apprendre progressivement les caractéristiques utiles des données. Dans le cas d'une image, un réseau peut apprendre à repérer des contours, puis des formes et des éléments plus complexes. Cela limite le travail nécessaire pour définir manuellement les caractéristiques à analyser. Cette approche est notamment utilisée pour traiter des images, du son et du texte. [3]

### Comparaison

Le Deep Learning étant une forme de Machine Learning, on le compare ici au ML classique, c'est-à-dire aux méthodes sans réseau profond. Les différences du tableau sont des tendances, elles dépendent du modèle et du problème traité. [2, 3, 4, 5]

| Critère | Machine Learning classique | Deep Learning |
| --- | --- | --- |
| Exemples de modèles | Régression linéaire, arbres de décision, forêts aléatoires | Réseaux de neurones profonds |
| Type de données | Souvent efficace sur des données organisées en tableau, comme un fichier CSV | Particulièrement adapté aux images, au son et au texte |
| Caractéristiques analysées | Leur sélection ou leur construction demande souvent un travail manuel | Le réseau apprend une grande partie des caractéristiques pendant l'entraînement |
| Quantité de données | Certains modèles donnent de bons résultats avec peu d'exemples | Entraîner un réseau profond à partir de zéro demande généralement beaucoup de données |
| Ressources nécessaires | Souvent moins coûteux pour les problèmes tabulaires courants | Les grands réseaux demandent davantage de calcul, souvent avec des cartes graphiques (GPU) |
| Compréhension des résultats | Un modèle simple, comme un petit arbre, peut être facile à expliquer | Les décisions d'un réseau profond sont généralement plus difficiles à expliquer |

### Quand choisir chaque approche ?

Pour des données en tableau, le ML classique constitue souvent un bon point de départ. C'est notamment le cas si le jeu de données est petit ou si les moyens de calcul sont limités. Pour estimer un prix à partir des caractéristiques d'un produit, on peut commencer par une régression ou une forêt d'arbres. Google recommande d'ailleurs les forêts de décision pour les données tabulaires. Si les résultats doivent être faciles à expliquer, un modèle simple peut aussi être préférable. [4, 5]

Le Deep Learning est particulièrement adapté aux tâches qui demandent d'apprendre des représentations complexes, comme reconnaître un objet dans une photo, transcrire de la parole ou traduire du texte. Le choix dépend aussi des ressources disponibles, car l'entraînement et l'exécution d'un grand réseau peuvent être coûteux. [2, 3]

Avoir peu de données n'exclut cependant pas le Deep Learning. Le transfert d'apprentissage permet de reprendre un modèle déjà entraîné sur une tâche proche, puis de l'adapter à ses propres données. Cela évite de repartir de zéro. [6]

Dans un projet, on peut donc commencer par une solution simple et évaluer ses résultats sur des données qui n'ont pas servi à l'entraînement. Passer à un modèle plus complexe se justifie si l'amélioration obtenue répond au besoin et compense le coût supplémentaire.

### Sources

Sources consultées le 28 octobre 2026 :

1. [Google for Developers : What is Machine Learning?](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml)
2. [IBM : What is deep learning?](https://www.ibm.com/think/topics/deep-learning)
3. [IBM : AI vs. machine learning vs. deep learning vs. neural networks](https://www.ibm.com/think/topics/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks)
4. [Google for Developers : Appropriate data for decision forests](https://developers.google.com/machine-learning/decision-forests/appropriate-data)
5. [Google for Developers : Introduction to decision forests](https://developers.google.com/machine-learning/decision-forests)
6. [Keras : Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/)

## 2. Faites une recherche sur les différentes applications du Deep Learning et présentez-en trois.

Le Deep Learning peut servir à reconnaître des informations, à créer du contenu ou à faire des prédictions scientifiques. Les trois exemples suivants illustrent ces usages dans des domaines différents.

### La reconnaissance de dessins avec Quick, Draw!

Quick, Draw! est une expérience de Google dans laquelle le joueur doit dessiner un objet en moins de vingt secondes. Un réseau de neurones essaie de reconnaître le dessin pendant sa réalisation. Par exemple, lorsqu'on dessine un vélo, le modèle doit identifier la catégorie correspondante malgré les différences de style entre les joueurs. [7]

Il s'agit d'une tâche de classification : le modèle associe un dessin à une catégorie apprise à partir d'exemples. L'intérêt est de reconnaître des formes sans écrire à la main une règle pour chaque façon de dessiner un objet. Google partage aussi les dessins collectés pour soutenir la recherche en apprentissage automatique. [7]

Cette expérience montre également les limites de la reconnaissance : un croquis ambigu ou inhabituel peut être mal classé. Le modèle peut proposer plusieurs interprétations, sans que la bonne réponse soit nécessairement la première. [7]

### La génération de musique avec Music Transformer

Music Transformer, présenté par l'équipe Magenta de Google, est un réseau de neurones capable de générer des séquences musicales. Il peut notamment prolonger un début de morceau fourni par l'utilisateur. C'est une application du Deep Learning à la création de contenu. [8]

Son architecture repose sur un Transformer et un mécanisme d'attention. Ce mécanisme lui permet de tenir compte des événements musicaux précédents pour produire la suite, notamment en reprenant ou en faisant varier un motif. Il ne se contente donc pas de générer des notes indépendantes : il cherche à maintenir une cohérence sur la durée. [8]

Un musicien peut s'en servir pour explorer plusieurs suites possibles à une idée de départ. Les résultats restent à sélectionner et à retravailler : les auteurs présentent aussi des exemples contenant trop de répétitions, des passages peu remplis ou des changements brusques. [8]

### La prédiction de structures de protéines avec AlphaFold

AlphaFold, développé par Google DeepMind, utilise le Deep Learning pour prédire la structure en trois dimensions des protéines à partir de leur séquence d'acides aminés. La forme d'une protéine est importante pour comprendre son rôle et ses interactions dans un organisme. [9]

Le modèle apprend à partir de données biologiques, dont des structures connues, pour proposer une structure pour d'autres protéines. Cette application aide les chercheurs à formuler des hypothèses et à orienter leurs expériences plus rapidement. Les prédictions peuvent notamment contribuer à l'étude des mécanismes biologiques et à la recherche de médicaments. [9, 10]

Ces résultats restent des prédictions. Leur fiabilité varie selon les régions de la protéine, et une structure prédite ne suffit pas à démontrer une fonction biologique ou l'efficacité d'un médicament. Il faut donc tenir compte des scores de confiance et des vérifications expérimentales nécessaires. [10]

### Sources

Sources consultées le 28 octobre 2026 :

7. [Google : Quick, Draw!](https://quickdraw.withgoogle.com/)
8. [Google Magenta : Music Transformer, Generating Music with Long-Term Structure](https://magenta.tensorflow.org/music-transformer)
9. [Google DeepMind : AlphaFold](https://deepmind.google/science/alphafold/)
10. [Jumper et al., Nature (2021) : Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2)

# Partie 2 : Le perceptron

## 1. Qu'est-ce qu'un perceptron ? Quel est le lien avec un neurone biologique ?

Un perceptron est un modèle de neurone artificiel utilisé pour classer des données. Dans sa forme classique, il reçoit plusieurs valeurs en entrée, calcule leur somme pondérée, puis applique un seuil pour produire une réponse binaire : 0 ou 1. Ses poids et son biais sont appris à partir d'exemples dont on connaît la classe attendue. C'est donc un modèle d'apprentissage supervisé. Un perceptron seul ne constitue pas un modèle de Deep Learning. [11]

Il s'inspire de manière simplifiée du neurone biologique : des signaux arrivent par les dendrites et les synapses, sont intégrés par le neurone et peuvent provoquer un potentiel d'action transmis par l'axone. Dans le perceptron, les entrées représentent les signaux reçus, les poids représentent l'influence des connexions et la fonction seuil détermine la sortie. Cette analogie aide à comprendre le principe, mais le perceptron ne reproduit pas toute la dynamique électrique et chimique d'un vrai neurone. [12]

## 2. Quelle est sa fonction mathématique et son usage ? Définissez les termes de l'équation.

En entrée, le perceptron reçoit un vecteur de caractéristiques, aussi appelées features : $x = (x_1, x_2, \ldots, x_n)$. Pour classer des fleurs, ces valeurs peuvent être la longueur et la largeur du pétale. La cible $y$ ne fait pas partie des entrées du modèle : elle sert pendant l'entraînement à vérifier la prédiction. En sortie, le modèle renvoie une classe binaire, ici 0 ou 1. Certaines conventions utilisent plutôt -1 et 1.

On commence par calculer un score, puis on applique une fonction d'activation à seuil : [11]

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

$$
\hat{y} = f(z) =
\begin{cases}
1 & \text{si } z \geq 0 \\
0 & \text{si } z < 0
\end{cases}
$$

| Terme | Signification |
| --- | --- |
| $n$ | Nombre de caractéristiques d'un exemple |
| $x_i$ | Valeur de la caractéristique numéro $i$ |
| $w_i$ | Poids associé à cette caractéristique, qui règle sa contribution au score |
| $\sum$ | Somme des produits de chaque entrée par son poids |
| $b$ | Biais, qui permet de décaler la frontière de décision |
| $z$ | Score obtenu avant l'activation |
| $f$ | Fonction seuil qui transforme le score en classe |
| $\hat{y}$ | Classe prédite, ici 0 ou 1 |

À échelle comparable des entrées, un poids grand en valeur absolue rend le score plus sensible aux variations de la caractéristique correspondante. Un poids proche de zéro réduit son influence. Quand une caractéristique augmente, un poids positif fait monter le score vers la classe 1, tandis qu'un poids négatif le fait descendre vers la classe 0. L'influence dépend donc aussi des valeurs et de l'échelle des entrées.

Le biais décale le score indépendamment des entrées. Sans biais, le score vaut zéro lorsque toutes les entrées sont nulles : la frontière doit donc passer par l'origine. Le biais permet de déplacer cette frontière. À poids fixés, augmenter le biais facilite la prédiction de la classe 1.

On peut aussi écrire la condition sous la forme $\sum_i w_i x_i \geq \theta$, avec un seuil $\theta = -b$. Les deux écritures sont équivalentes. La sortie est une décision, pas une probabilité. [11, 13]

Le perceptron sert à séparer deux classes par une frontière définie par $\sum_i w_i x_i + b = 0$. Avec deux entrées, cette frontière est une droite ; avec davantage d'entrées, c'est un hyperplan. Il peut représenter les fonctions logiques ET et OU, mais pas XOR avec les entrées d'origine, car les classes de XOR ne sont pas séparables par une seule droite. [14]

Par exemple, pour représenter ET, on peut choisir $w_1 = w_2 = 1$ et $b = -1{,}5$. Avec les entrées $(1,1)$, le score vaut $0{,}5$, donc la sortie vaut 1. Pour les trois autres couples binaires, le score est négatif et la sortie vaut 0.

## 3. Donnez une ou plusieurs règles d'apprentissage du perceptron.

La règle classique du perceptron corrige les paramètres après une erreur de classification. Avec des classes codées 0 et 1, on calcule l'erreur $e = y - \hat{y}$, où $y$ est la réponse attendue. On met ensuite à jour chaque poids et le biais : [13, 15]

$$
w_i^{\text{nouveau}} = w_i^{\text{ancien}} + \eta e x_i
$$

$$
b^{\text{nouveau}} = b^{\text{ancien}} + \eta e
$$

Le paramètre $\eta > 0$ est le taux d'apprentissage : il règle l'amplitude des corrections. Les mises à jour utilisent toutes la même erreur, calculée avant de modifier les paramètres. [13, 15]

- Si la prédiction est correcte, $e = 0$ : rien ne change.
- Si le modèle prédit 0 au lieu de 1, $e = 1$ : on ajoute $\eta x_i$ à chaque poids et $\eta$ au biais.
- S'il prédit 1 au lieu de 0, $e = -1$ : on soustrait ces mêmes quantités.

Multiplier par $x_i$ adapte la correction à l'entrée concernée. Si $x_i = 0$, son poids ne change pas. Plus sa valeur absolue est grande, plus la correction est importante. Le signe compte aussi : ajouter $\eta x_i$ diminue le poids si $x_i$ est négatif. Tous les poids n'augmentent donc pas forcément après une prédiction 0 au lieu de 1.

Sur le même exemple, la variation du score après la correction vaut $\Delta z = \eta e(\sum_i x_i^2 + 1)$. Elle est positive si le modèle a prédit 0 au lieu de 1, et négative dans le cas inverse. La correction pousse ainsi le score dans le sens attendu, sans forcément suffire à corriger l'erreur en une seule mise à jour.

On peut choisir un taux comme 0,01 ou 0,1. Pour le perceptron classique initialisé à zéro, changer ce taux positif constant multiplie tous les paramètres par le même facteur, à ordre des exemples identique et en arithmétique exacte. Les décisions restent alors les mêmes : un taux plus petit ne signifie donc pas forcément un apprentissage plus lent.

On initialise les paramètres, par exemple à zéro, puis on parcourt les exemples en appliquant cette règle. Un passage complet sur le jeu d'entraînement s'appelle une époque. On recommence jusqu'à une époque sans erreur ou jusqu'au nombre maximal d'époques fixé. [13]

Pour un jeu fini linéairement séparable, la règle classique, avec un taux positif constant et des passages répétés sur tous les exemples, converge vers une séparation correcte. Si les données ne sont pas linéairement séparables, cette convergence n'est pas garantie : augmenter le nombre d'époques ne suffit pas à résoudre cette limite. [13, 14]

## 4. Le perceptron utilise généralement une fonction d'activation, laquelle ?

Le perceptron classique utilise une fonction seuil, aussi appelée fonction échelon ou fonction de Heaviside. Dans la convention retenue ici, elle renvoie 1 si la somme pondérée avec le biais est positive ou nulle, et 0 sinon. Elle transforme donc le score en décision binaire. La formule est celle présentée à la question 2. [14]

Il faut distinguer cette activation des fonctions sigmoïde ou ReLU utilisées dans d'autres modèles de neurones artificiels. Le perceptron classique conserve une activation à seuil.

## 5. Quel est le processus d'entraînement du perceptron ?

L'entraînement suit les étapes suivantes : [13, 14]

1. Préparer les exemples et leurs étiquettes, puis réserver des données pour le test.
2. Initialiser les poids et le biais, ici à zéro, et choisir le taux d'apprentissage ainsi qu'un nombre maximal d'époques.
3. Pour chaque exemple d'entraînement, calculer la somme pondérée et appliquer la fonction seuil.
4. Comparer la prédiction à l'étiquette attendue, puis corriger les poids et le biais avec la règle de la question 3.
5. Recommencer les passages sur les données jusqu'à une époque sans erreur ou jusqu'à la limite fixée.
6. Évaluer le modèle final sur les données de test, sans mettre à jour ses paramètres.

Séparer l'entraînement du test permet de vérifier si le modèle classe aussi des exemples qu'il n'a pas utilisés pour apprendre. L'exactitude correspond au nombre de bonnes prédictions divisé par le nombre d'exemples évalués.

## 6. Quelles sont les limites du perceptron ?

Sa principale limite est sa frontière de décision linéaire. Un perceptron seul ne peut pas séparer des classes qui nécessitent une frontière plus complexe, comme XOR avec ses deux entrées d'origine. [14]

Le XOR illustre cette limite :

| $x_1$ | $x_2$ | Classe XOR |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Les points de classe 1 occupent une diagonale du carré et ceux de classe 0 l'autre. Aucune droite ne sépare ces deux classes. Avec des passages répétés sur ces quatre exemples, le perceptron continue donc à faire des erreurs : il ne peut pas terminer une époque sans erreur. Un réseau à plusieurs couches avec des activations non linéaires peut représenter XOR et des frontières plus complexes.

Si les classes se chevauchent ou si des étiquettes sont contradictoires, il ne peut pas obtenir une séparation parfaite. L'apprentissage peut alors continuer à corriger les paramètres sans converger, d'où l'intérêt de fixer un nombre maximal d'époques. [13]

Dans sa forme binaire classique, la sortie se limite à deux classes et ne fournit pas de probabilité. Le modèle n'apprend pas non plus de représentations intermédiaires comme un réseau à plusieurs couches. Il reste donc adapté à des problèmes simples dont les caractéristiques permettent une séparation linéaire. [11, 14]

## 7. Développer un perceptron en Python orienté objet et le tester sur des données aléatoires.

La classe `Perceptron` est écrite dans `perceptron.py` et testée dans `perceptron.ipynb`.

### Conception de la classe

- `__init__(lr, n_epochs)` enregistre les deux hyperparamètres : le taux d'apprentissage et le nombre maximal d'époques. Les poids et le biais sont des paramètres appris, créés dans `fit`.
- `fit(X, y)` initialise les poids à zéro, puis parcourt les exemples à chaque époque. Pour chaque exemple, elle applique la règle de correction $w \leftarrow w + \eta\,(y - \hat{y})\,x$ et $b \leftarrow b + \eta\,(y - \hat{y})$. Elle enregistre le nombre d'erreurs de chaque époque dans `errors_`. [15]
- `predict(X)` calcule la somme pondérée $z = w \cdot x + b$ et renvoie 1 si $z \geq 0$, sinon 0.
- `score(X, y)` renvoie l'exactitude, c'est-à-dire la part de prédictions correctes.

### Tests

Premier test : 100 points tirés au hasard dans le plan, de classe 1 lorsque $x_1 + x_2 > 0$. Une droite sépare les deux classes. Le nombre d'erreurs par époque est 14, 10, puis 0 à partir de la troisième époque, et l'exactitude atteint 1,0. Les poids appris sont proches l'un de l'autre (environ 0,37 et 0,40) avec un biais nul, ce qui retrouve la règle utilisée pour créer les données.

Second test : le problème XOR, qui reprend la limite décrite à la question 6. Le nombre d'erreurs ne tombe jamais à 0 (entre 3 et 4 par époque) et l'exactitude reste à 0,5. Le perceptron corrige un point en en déréglant un autre, car aucune droite ne sépare les deux classes.

### Sources

Sources consultées le 28 octobre 2026 :

11. [Cornell Virtual Workshop : The Basic Building Block, The Perceptron](https://cvw.cac.cornell.edu/SciML/mlp/perceptron)
12. [Stanford : Neural Networks, Neuron](https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/Neuron/index.html)
13. [Chris Pollett, San José State University : Perceptron Learning](https://www.cs.sjsu.edu/~pollett/256.2.21f/Lec20210908.html)
14. [University of Wisconsin-Madison : CS 540 Lecture Notes, Neural Networks](https://pages.cs.wisc.edu/~dyer/cs540/notes/nn.html)
15. [Cornell : Introduction to Neural Networks, règle d'apprentissage du perceptron](https://www.cs.cornell.edu/courses/cs4700/2017fa/lectures/pdf/CS4700-15-ML_Neural_Nets_intro_cpg_v6.pdf)
