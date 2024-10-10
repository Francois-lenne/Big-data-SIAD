# Présentation du projet 


## Contexte 

Dans le cadre de notre projet Big data du master SIAD, nous participons à la compétition Kaggle Tweet Disaster. Notre objectif est de définir si un tweet concerne réellement un événement ou non. Pour ce faire, nous avons utilisé les méthodes lié au domaine du NLP (Natural Language Process). Pour retraiter les tweet (lématisations ...). Puis la création de variable avec les mots  les plus utilisés. Nous avons aussi utilisé des package comme Geotext qui utilise des modèles de machine learning pour définir si le lieux est une ville, pays ... Ces packages nous permettront ainsi d'enrichir notre jeu de donnée.

Notre projet est compsé de **4 membres** :

* [Louis Toulouse](https://www.linkedin.com/in/louis-toulouse/)
* [François Lenne](https://www.linkedin.com/in/fran%C3%A7ois-lenne-5975b9174/)
* [Yoann Playez](https://www.linkedin.com/in/yoann-playez-075ab7207/)
* [Ronan Patin](https://www.linkedin.com/in/ronan-patin-186aab192/)

Le projet est encadré par deux enseignant chercheur de l'université de Lille :


* [Maxime MORGE](http://www.lifl.fr/~morge), CRIStAL/ULille

* [Virginie Deslart](https://www.linkedin.com/in/virginie-delsart-9a45b81b9/?originalSubdomain=fr), Clerse/ULille


## Stack technique du projet :computer:

[![My Skills](https://go-skill-icons.vercel.app/api/icons?i=py,md,git,github,vscode,regex,html,css,js,fastapi,bootstrap)](https://skills.thijs.gg)

# Installation

Pour utiliser ce projet, vous devez avoir Python 3 installé sur votre ordinateur. Vous pouvez télécharger Python 3 à partir du site web officiel de Python.

***
Clonez ce dépôt de code à l'aide de la commande git clone <https://github.com/Francois-lenne/Big-data-SIAD.git> dans votre terminal.
```
git clone https://github.com/Francois-lenne/Big-data-SIAD.git

```

Allez dans le répertoire (le chemin peut varier selon vos répertoires) du projet à l'aide de la commande : 
```
cd ~/GitHub/Big-data-SIAD/api

```

Installez les dépendances python en utilisant la commande :
```
pip install -r requirements.txt

```
***

# Utilisation

Une fois que vous avez installé les dépendances, vous pouvez utiliser ce projet Python en suivant les instructions suivantes :

***

Ouvrez votre terminal et accédez au répertoire du projet (le chemin peut varier selon vos répertoires).
```
cd ~/GitHub/Big-data-SIAD/api

```

Lancez l'entraînement du modèle avec la commande :

```
python prepare.py

```

Exécutez la commande suivante :

```
uvicorn --reload main:app

```

L'application est lancé, vous pouvez ouvrir dans un navigateur web le fichier __app.html__ et soumettre vos tweets.

***


# Contributeurs

## Développements réalisés par Ronan 

- :man_technologist: Développements du **front-end** du site web
- :pilot: Gestion de projet (trello, répartition des tâches)
- :bookmark_tabs: Rédaction du rapport
- :chart_with_upwards_trend: Modélisations

## Développements réalisés par Yoann  

- :man_technologist: Développements du **back end** du site web
- :bookmark_tabs: Rédaction du rapport
- :chart_with_upwards_trend: Modélisations
- :globe_with_meridians: Déploiements du site web

## Développements réalisés par François  

- :construction_worker: Feature engineering
- :bookmark_tabs: Rédaction du rapport
- :chart_with_upwards_trend: Modélisations
- :globe_with_meridians: Déploiements du site web


## Développements réalisés par Louis  

- :construction_worker: Feature engineering
- :bookmark_tabs: Rédaction du rapport
- :chart_with_upwards_trend: Modélisations
- :construction: Test de plusieurs modélisations

## Package utilisé pour réaliser les retraitements et le modèle


* [Pandas](https://pandas.pydata.org/) {Version 1.5.3}
* [Numpy](https://numpy.org/) {Version 1.24.2}
* [re](https://docs.python.org/3/library/re.html) {Version 3.11.2 }
* [spacy](https://spacy.io/usage) {Version 3.5}
* [sympi](https://www.sympy.org/en/index.html) {Version 1.7.1}
* [geotext](https://pypi.org/project/geotext/) {Version 0.4.0}
* [Sklearn](https://scikit-learn.org/stable/) {Version 1.2.2}
* [FastAPI](https://fastapi.tiangolo.com/) {Version 0.89.1}
* [joblib](https://joblib.readthedocs.io/en/latest/) {Version 1.2.0}
* [nltk](https://www.nltk.org/) {Version 3.8.1}
* [pydantic](https://docs.pydantic.dev/) {Version 1.10.4}
* [unvicorn](https://www.uvicorn.org/) {Version 0.21.1}


# Licence
Ce projet est sous licence MIT. Consultez le fichier LICENSE.txt pour plus d'informations.




  
