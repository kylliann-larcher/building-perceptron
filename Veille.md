# Veille technologique — Machine Learning, Deep Learning et perceptron

**Projet :** Building Perceptron  
**Dernière consultation des sources :** 28 octobre 2026

> **Objectif :** comprendre comment un perceptron apprend à classer des données, identifier ses limites et relier ces connaissances au projet Breast Cancer Wisconsin.

## Sommaire

1. [Machine Learning et Deep Learning](#1-machine-learning-et-deep-learning)
2. [Trois applications du Deep Learning](#2-trois-applications-du-deep-learning)
3. [Le fonctionnement du perceptron](#3-le-fonctionnement-du-perceptron)
4. [Son apprentissage](#4-son-apprentissage)
5. [Ses limites](#5-ses-limites)
6. [Application au projet BCW](#6-application-au-projet-bcw)
7. [Les évolutions à connaître](#7-les-évolutions-à-connaître)
8. [Conclusion](#8-conclusion)
9. [Sources](#9-sources)

---

## 1. Machine Learning et Deep Learning

### Le Machine Learning

Le **Machine Learning** (ML), ou apprentissage automatique, permet à un programme d'apprendre des relations à partir de données. Le modèle utilise ensuite ces relations pour produire des prédictions sur de nouveaux exemples.[^ml]

On distingue notamment :

| Apprentissage | Informations fournies | Exemple |
| --- | --- | --- |
| **Supervisé** | Des exemples et leurs réponses connues | Classer des observations en deux catégories |
| **Non supervisé** | Des exemples sans réponse cible | Chercher des groupes ou des axes de variation |
| **Par renforcement** | Des interactions et des récompenses | Apprendre une stratégie dans un environnement |

Dans l'apprentissage supervisé, la **classification** prédit une catégorie et la **régression** prédit une valeur numérique. Le projet BCW est un problème de classification.

### Le Deep Learning

Le **Deep Learning** (DL), ou apprentissage profond, fait partie du Machine Learning. Il utilise des réseaux de neurones à plusieurs couches pour apprendre des représentations et des relations complexes.[^mlp]

Un perceptron seul est un neurone artificiel de classification : il ne constitue pas un réseau profond.

### Comment choisir ?

Les différences ci-dessous sont des tendances, pas des règles absolues.

| Critère | ML classique | Deep Learning |
| --- | --- | --- |
| Modèles courants | Modèles linéaires, arbres, forêts | Réseaux de neurones à plusieurs couches |
| Données fréquentes | Tableaux de mesures | Images, son, texte |
| Caractéristiques | Sélection ou construction souvent nécessaire | Une partie des représentations est apprise |
| Ressources | Souvent adaptées à un petit projet tabulaire | Les grands réseaux peuvent être coûteux |
| Explication | Certains modèles simples sont faciles à examiner | Les décisions peuvent être plus difficiles à interpréter |

Google présente les forêts de décision comme une référence pertinente pour les tableaux. Cela ne garantit pas qu'elles soient meilleures sur chaque jeu de données.[^forests]

**Pour ce projet :** le perceptron est imposé par le sujet. Il faut comprendre son fonctionnement, puis juger ses résultats avec un protocole adapté.

---

## 2. Trois applications du Deep Learning

### Quick, Draw! — reconnaître un dessin

Le joueur dessine un objet et un réseau tente d'en reconnaître la catégorie. Le modèle apprend à partir de nombreuses manières de représenter les mêmes objets.[^quickdraw]

- **Usage :** classification d'une entrée visuelle.
- **Limite :** un dessin ambigu ou inhabituel peut être mal reconnu.
- **Lien avec le projet :** comparer la prédiction à une réponse attendue pour mesurer les erreurs.

### Music Transformer — générer de la musique

Présenté par Magenta le **13 décembre 2018**, Music Transformer utilise l'attention pour générer une performance musicale ou prolonger un motif. Il cherche à conserver une cohérence sur de longues séquences.[^music]

- **Usage :** génération de contenu.
- **Limite :** les auteurs montrent des répétitions excessives et des changements brusques.
- **Lien avec le projet :** choisir des critères d'évaluation correspondant à la tâche.

### AlphaFold — prédire des structures biomoléculaires

AlphaFold aide à prédire des structures. Le **8 mai 2024**, AlphaFold 3 étend cette approche aux complexes biomoléculaires et à leurs interactions. Un addendum de Nature du **27 novembre 2024** annonce la mise à disposition du code d'inférence.[^alphafold][^af3]

- **Usage :** prédiction scientifique pour orienter des recherches.
- **Limite :** une structure prédite s'interprète avec ses incertitudes et son contexte expérimental.
- **Lien avec le projet :** distinguer une prédiction d'un résultat effectivement vérifié.

Ces applications illustrent trois usages : **reconnaître, générer et prédire une structure**. Quick, Draw! et Music Transformer sont des exemples établis ; ils ne sont pas présentés comme des nouveautés de 2026.

---

## 3. Le fonctionnement du perceptron

### Un modèle de neurone artificiel

Le perceptron reçoit plusieurs nombres, calcule une somme pondérée et applique un seuil. Dans la version binaire étudiée, sa sortie vaut **0 ou 1**.[^perceptron]

L'analogie avec le neurone biologique est simplifiée :

| Élément biologique | Représentation dans le perceptron |
| --- | --- |
| Signaux reçus | Entrées |
| Influence des connexions | Poids |
| Intégration des signaux | Somme pondérée |
| Production d'une sortie | Décision par le seuil |

Ce modèle ne reproduit pas toute la dynamique électrique et chimique d'un neurone.

### Le calcul

On calcule d'abord un **score** :

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

Puis on applique la fonction seuil, avec la convention :

$$
\hat{y} =
\begin{cases}
1 & \text{si } z \geq 0 \\
0 & \text{si } z < 0
\end{cases}
$$

| Symbole | Signification |
| --- | --- |
| $x_i$ | Valeur d'une caractéristique |
| $w_i$ | Poids associé à cette caractéristique |
| $b$ | Biais, qui décale la frontière de décision |
| $n$ | Nombre de caractéristiques |
| $z$ | Score calculé |
| $\hat{y}$ | Classe prédite |

Avec deux caractéristiques, la frontière est une **droite** lorsque les poids ne sont pas tous nuls. Avec davantage de caractéristiques, c'est un **hyperplan**.

Le score est un résultat de calcul, **pas une probabilité**.[^perceptron]

### Exemple : la fonction logique ET

On choisit deux poids égaux à 1 et un biais de −1,5.

| Entrées | Calcul du score | Sortie |
| --- | --- | --- |
| (0, 0) | 0 + 0 − 1,5 = −1,5 | 0 |
| (0, 1) | 0 + 1 − 1,5 = −0,5 | 0 |
| (1, 0) | 1 + 0 − 1,5 = −0,5 | 0 |
| (1, 1) | 1 + 1 − 1,5 = 0,5 | 1 |

La sortie vaut 1 uniquement lorsque les deux entrées valent 1 : le modèle représente bien ET.

### La fonction d'activation

Le perceptron classique utilise une **fonction seuil**, aussi appelée fonction échelon ou Heaviside dans la convention retenue.

La sigmoïde et ReLU sont utilisées dans d'autres modèles, notamment les réseaux multicouches.[^mlp]

---

## 4. Son apprentissage

### Le principe

Le modèle compare sa prédiction à la bonne réponse. S'il se trompe, il corrige les poids et le biais.

L'erreur est :

$$
e = y - \hat{y}
$$

La règle de correction est :

$$
\begin{aligned}
w_i &\leftarrow w_i + \eta e x_i \\
b &\leftarrow b + \eta e
\end{aligned}
$$

- $y$ est la réponse attendue.
- $\hat{y}$ est la réponse prédite.
- $\eta > 0$ est le **taux d'apprentissage**, qui règle la taille des corrections.

Toutes les mises à jour d'un exemple utilisent la même erreur, calculée avant de modifier les paramètres.

| Situation | Erreur | Effet |
| --- | --- | --- |
| Prédiction correcte | 0 | Aucun changement |
| Prédit 0, attendu 1 | +1 | Ajouter $\eta x_i$ aux poids et $\eta$ au biais |
| Prédit 1, attendu 0 | −1 | Soustraire ces quantités |

### Un exemple de correction

Pour les valeurs suivantes :

- entrées : $x=[0{,}8;0{,}8]$ ;
- poids : $w=[0{,}2;0{,}1]$ ;
- biais : $b=-0{,}3$ ;
- réponse attendue : $y=1$.

Le score initial vaut :

$$
0{,}2 \times 0{,}8 + 0{,}1 \times 0{,}8 - 0{,}3 = -0{,}06
$$

Le modèle prédit 0 : l'erreur vaut donc 1. Avec un taux de 0,1, les poids deviennent **[0,28 ; 0,18]** et le biais **−0,2**. Le nouveau score vaut **0,168** : la sortie devient 1.

Une correction peut aussi modifier la réponse sur d'autres observations. Les erreurs ne diminuent donc pas obligatoirement à chaque passage.

### Les étapes de l'entraînement

1. Préparer les observations et leurs réponses.
2. Réserver des données pour l'évaluation.
3. Initialiser les poids et le biais.
4. Parcourir les exemples, prédire et corriger en cas d'erreur.
5. Répéter jusqu'à une époque sans correction ou une limite fixée.
6. Évaluer le modèle sur les observations gardées à part.

Une **époque** correspond à un passage complet sur les exemples d'entraînement. Il faut distinguer les corrections pendant ce passage des erreurs du modèle final à la fin du passage.

---

## 5. Ses limites

### Une frontière linéaire

Le perceptron ne peut pas séparer toutes les configurations de données. Le problème **XOR** en donne un exemple :

| Entrée 1 | Entrée 2 | Réponse XOR |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Les deux points de classe 1 sont opposés en diagonale. **Aucune droite ne les sépare parfaitement des deux points de classe 0.**

Augmenter le nombre d'époques ne change pas cette limite dans les entrées d'origine. Un réseau multicouche avec des activations non linéaires, ou une transformation adaptée des caractéristiques, peut modifier la capacité de représentation.[^mlp]

### Les conséquences pratiques

- Une séparation parfaite n'est pas garantie lorsque les classes se chevauchent.
- Atteindre le maximum d'époques ne signifie pas avoir convergé.
- La version étudiée fournit deux classes, sans probabilité native.
- Elle n'apprend pas de représentations intermédiaires comme un réseau multicouche.

---

## 6. Application au projet BCW

### Les données

Le jeu **Breast Cancer Wisconsin Diagnostic** contient **569 observations et 30 caractéristiques numériques** calculées à partir d'images de noyaux cellulaires. La cible distingue les catégories **B — bénin** et **M — malin**.[^bcw]

Le projet traite les mesures extraites sous forme de tableau. Dans le fichier, il faut distinguer les caractéristiques, la cible `diagnosis`, l'identifiant et les éventuelles colonnes vides d'export.

### La démarche à suivre

| Étape | Travail à réaliser | Point de vigilance |
| --- | --- | --- |
| **Nettoyer** | Vérifier types, valeurs manquantes, doublons et colonnes inutiles | Justifier chaque modification |
| **Explorer** | Observer distributions, valeurs extrêmes et corrélations | Une corrélation ne prouve pas une cause |
| **Séparer** | Réserver le test et prévoir une validation | Ne pas choisir les réglages sur le test |
| **Standardiser** | Rendre les échelles comparables | Apprendre les paramètres sur le train |
| **Réduire la dimension** | Choisir et justifier une méthode, par exemple l'ACP | Variance expliquée et performance sont différentes |
| **Modéliser** | Utiliser la classe Perceptron développée | Conserver les graines et paramètres |
| **Évaluer** | Mesurer les résultats et examiner les erreurs | Comparer à une référence simple |

L'**ACP**, ou PCA, construit de nouveaux axes expliquant une partie de la variation des caractéristiques. Conserver 95 % de variance ne signifie pas obtenir 95 % de bonnes prédictions.[^pca]

Une **Pipeline** aide à ajuster les transformations uniquement sur l'entraînement de chaque pli de validation, puis à les appliquer sans réajustement aux observations évaluées.[^pitfalls]

### Les métriques à présenter

Convention proposée : **M = 1**, classe positive, et **B = 0**.

| Indicateur | Interprétation |
| --- | --- |
| **Accuracy** | Proportion totale de prédictions correctes |
| **Précision** | Parmi les M prédits, proportion réellement M |
| **Rappel** | Parmi les vrais M, proportion détectée |
| **F1** | Compromis entre précision et rappel |
| **Matrice de confusion** | Répartition des bonnes réponses et des types d'erreurs |

Un **faux négatif** correspond alors à un M prédit B. La classe positive et le traitement des divisions par zéro doivent être précisés. Une référence naïve majoritaire aide à interpréter l'accuracy.[^metrics]

> **Périmètre du document :** cette veille pose les bases théoriques. La classe, les tests et les résultats expérimentaux doivent être présentés dans le script et le notebook. Un score sur ce jeu d'étude ne démontre pas à lui seul une validité clinique.

---

## 7. Les évolutions à connaître

Les éléments suivants ouvrent la discussion. Ils ne remplacent pas l'implémentation demandée par le sujet.

### KAN — ICLR 2025

Les **Kolmogorov-Arnold Networks** proposent des fonctions apprenables sur les connexions, notamment paramétrées par des splines. Les auteurs rapportent des résultats sur des tâches scientifiques de petite taille et d'approximation de fonctions.[^kan]

**À retenir :** c'est une piste de recherche sur la représentation des fonctions. Les auteurs signalent aussi un entraînement lent ; leurs résultats ne prouvent pas une supériorité universelle.

### TabPFN — Nature, 8 janvier 2025

**TabPFN** est un modèle tabulaire préentraîné sur des jeux synthétiques. L'étude de Hollmann et ses coauteurs présente notamment des évaluations jusqu'à 10 000 observations et 500 caractéristiques.[^tabpfn]

**À retenir :** le préentraînement concerne aussi les tableaux. Un bénéfice sur BCW doit être vérifié avec un protocole adapté.

### TabPFN-3.5 — 15 septembre 2026

Prior Labs annonce **TabPFN-3.5** et revendique une première place sur TabArena et BeyondArena, ainsi qu'une variante Fast en alpha.[^tabpfn35]

**À retenir :** il s'agit d'une annonce du concepteur. Avant un essai, examiner la version, le protocole, les ressources et les conditions d'accès. Une position sur un benchmark ne garantit pas le meilleur résultat sur chaque jeu.

### Ce que cela change pour le projet

| Priorité | Sujet | Utilisation |
| --- | --- | --- |
| **Maintenant** | Perceptron, préparation et évaluation | Implémenter et expliquer |
| **Approfondissement** | MLP et non-linéarité | Expliquer les limites et proposer une piste |
| **Ouverture** | KAN et TabPFN | Présenter les travaux et leurs limites |

L'énoncé demande de **proposer les solutions supplémentaires sans les implémenter**.

---

## 8. Conclusion

Le perceptron permet de comprendre comment un modèle transforme des mesures en une décision et corrige ses paramètres à partir d'exemples. Son fonctionnement est explicable, mais sa frontière linéaire limite les problèmes qu'il peut résoudre.

Pour **Building Perceptron**, l'essentiel est de relier la théorie au code, de préparer les données sans fuite et d'interpréter plusieurs métriques. Les évolutions récentes enrichissent la discussion ; elles ne dispensent pas d'une évaluation rigoureuse.

## 9. Sources

Les références ci-dessous ont été consultées le **28 octobre 2026**. Les documentations décrivent les outils ; les publications rapportent les expériences de leurs auteurs ; les annonces des concepteurs sont identifiées comme telles.

### Documents du projet

- [Énoncé Building Perceptron — commit examiné](https://github.com/kylliann-larcher/building-perceptron/blob/56e4ceefd3bc99a6b2cbc544c8d6f22298a3f7a6/Sujet/Building%20Perceptron.pdf)
- [Données du projet](data/raw/bcw_data.csv)
- [Descriptif des données](data/raw/bcw_description.txt)

### Références documentaires

[^ml]: Google for Developers. [What is Machine Learning?](https://developers.google.com/machine-learning/intro-to-ml/what-is-ml)
[^mlp]: scikit-learn. [Neural network models (supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
[^forests]: Google for Developers. [Appropriate data for decision forests](https://developers.google.com/machine-learning/decision-forests/appropriate-data)
[^quickdraw]: Google. [Quick, Draw!](https://quickdraw.withgoogle.com/)
[^music]: Huang, Simon et Dinculescu. [Music Transformer: Generating Music with Long-Term Structure](https://magenta.tensorflow.org/music-transformer), 13 décembre 2018.
[^alphafold]: Google DeepMind. [AlphaFold — présentation et chronologie](https://deepmind.google/science/alphafold/)
[^af3]: Nature. [Addendum: Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-08416-7), 27 novembre 2024.
[^perceptron]: scikit-learn. [Perceptron — documentation de l'API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
[^bcw]: UCI Machine Learning Repository. [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), DOI : 10.24432/C5DW2B.
[^pca]: scikit-learn. [PCA — documentation de l'API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
[^pitfalls]: scikit-learn. [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
[^metrics]: scikit-learn. [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
[^kan]: Liu et al. [KAN: Kolmogorov–Arnold Networks](https://proceedings.iclr.cc/paper_files/paper/2025/hash/afaed89642ea100935e39d39a4da602c-Abstract-Conference.html), ICLR 2025.
[^tabpfn]: Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6), Nature, 637, 319–326, 8 janvier 2025.
[^tabpfn35]: Prior Labs. [TabPFN-3.5: Technical Report](https://priorlabs.ai/technical-reports/tabpfn-3-5), 15 septembre 2026.

