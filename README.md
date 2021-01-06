# Projet d'ingénierie logicielle pour l'IA

### Exercice 1

Récupérer le docker du stagiaire :
```docker pull wiidiiremi/projet_industrialisation_ia_3a```

Lancer le docker (port 8080)
```docker run -p 8080:8080 wiidiiremi/projet_industrialisation_ia_3a```

### Installation de Spacy

TODO


### Documentation service endpoint

Le service endpoint est accessible via plusieurs page. 
  - Tout d'abord une page d'accueil qui n'a pas de réel intérêt si ce n'est de prévenir l'utilisateur que le service est bien lancé
  - Une page de requête. Cette page est la page principale. Elle est accessible via un navigateur à l'adresse suivante : [adresse réseau (exemple : 127.0.0.1)]:[port]/api/intent?sentence=[phrase à classer]. Noter que la phrase à classer doit être entourée de **"**. Le retour de la page est alors un dictionnaire qui contient pour chaque intent un facteur de confiance, assimilable à une probabilité (assimilable seulement car la somme ne vaut pas forcément 1, témoignant du fait que des phrases puisse à la fois présenter un aspect find-train et purchase par exemple).  
