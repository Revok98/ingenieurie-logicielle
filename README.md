# Projet d'ingénierie logicielle pour l'IA

### Lancement de l'image Docker
* Déplacement dans le répertoire app :

```cd app```

* Construction de l'image :

```docker build -t app .```

* Lancement l'image docker (port 8080)

```docker run --net=host app```

* Enjoy =D


### Documentation service endpoint

Le service endpoint est accessible via plusieurs pages. 
  - Tout d'abord une page d'accueil qui n'a pas de réel intérêt si ce n'est de prévenir l'utilisateur que le service est bien lancé
  - Une page de requête. Cette page est la page principale. Elle est accessible via un navigateur à l'adresse suivante : [adresse réseau (exemple : 127.0.0.1)]:[port]/api/intent?sentence=[phrase à classer]. Noter que la phrase à classer doit être entourée de **"**. Le retour de la page est alors un dictionnaire qui contient pour chaque intent un facteur de confiance, assimilable à une probabilité (assimilable seulement car la somme ne vaut pas forcément 1, témoignant du fait que des phrases puisse à la fois présenter un aspect find-train et purchase par exemple).  

### Documentation organisation github

* app : dossier qui contient les éléments nécessaires à la création de l'image Docker

* data : Dossier qui contient les datasets de training, testing et les résultats de prédictions avec le modèle fourni (predict.json) et notre modèle (predict2.json)

* images : contient des images qui sont à visualiser directement dans le notebook reponse_exercices.ipynb

* utils : Contient des fonctions utiles à la visualisation et à la réponse aux exercices dans deux fichiers python.

* Reponse_exercices.ipynb : Jupyter notebook qui contient les différentes réponses aux exercices et les visualisations nécessaires pour appuyer les réponses.

* stat_dataset.ipynb : fichier qui permet la visualisation de statistiques sur le dataset d'entraînement et de test

* unitary_test.ipynb : Permet d'effectuer des tests unitaires sur les fonctions de statistiques du dataset
