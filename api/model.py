# Les librairies importées :
# - joblib permet de sauvegarder ou charger un objet python en fichier binaire
# - pandas permet la création et la manipulation des dataframe
# - os permet l'utilisation des fonctionnalités du système d'exploitation
# - sklearn permet la création de divers modèle de machine learning

import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Programmation orientée objet 
# L'objet TweetCheckerModel est un modèle NLP(Natural Language Processing) qui est destiné a être instancié par notre API.
# C'est une version adapté pour l'API pour plus de précision sur le modèle un notebook est disponible dans le projet.

class TweetCheckerModel:

    # Constructeur 
    # Les variables trained et imported permettent de vérifier si les données d'entraînements sont importées et que le modèle est entraîné.

    def __init__(self):
        self.trained = False
        self.imported = False

    # Méthode load
    # Vérification si des sauvegardes du modèle existent 
    # Cas 1 : elles n'existent pas, une erreur est renvoyée invitant à lancer le script d'entraînement prepare.py.
    # Cas 2 : elles existent, les sauvegardes sont chargés puis les objets sont instanciés
    def load(self):
        if os.path.exists('model.gz') == False or os.path.exists('classifier.gz') == False:
            raise Exception("Aucune sauvegarde du modèle a été trouvé, vous devez lancer le script prepare.py")
        else:
            self.model = joblib.load('model.gz')
            self.classifier = joblib.load('classifier.gz')
            self.trained = True

    # Méthode data
    # Tentative d'importation du jeu de données (format CSV avec séparateur ",").
    # Cas 1 : le jeu de données est importé avec succès, la variable imported prend la valeur True.
    # Cas 2 : le jeu de données n'a pas pu être importé, une erreur est renvoyée.
    def data(self,url):
        try:
            self.train = pd.read_csv(url, sep=',')
            self.imported = True
        except:
            raise Exception("Les données n'ont pas pu être importées")
    # Méthode prepare
    # Préparation du jeu de données importés, création du modèle et entraînement.
    # Cas 1 : Les données d'entraînement ne sont pas importées, une erreur est renvoyée.
    # Cas 2 : Les données d'entraînement sont importées, les données sont préparées, les modèles sont créés et entraînés.
    def prepare(self):
        if not self.imported == True:
            raise Exception("La préparation est impossible car les données non pas été encore importées")
        else:
            self.classifier=TfidfVectorizer()
            self.model = SVC(probability=True)
            self.train['text_cleaned']=self.train['text'].apply(clean)
            x=self.train['text_cleaned'].values
            y=self.train['target'].values
            x=self.classifier.fit_transform(x)
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.10,random_state=44,stratify=y)
            logReg=LogisticRegression(penalty='l2')
            logReg.fit(x_train,y_train)
            pred=logReg.predict(x_test)
            R=logReg.predict(x_train)
            self.model.fit(x_train,y_train)
            y_pre=self.model.predict(x_test)
            joblib.dump(self.model,'model.gz')
            joblib.dump(self.classifier,'classifier.gz')
            self.trained == True
            print("Le modèle a été entraîné et sauvegardé")
    
    # Méthode submit
    # Soumission d'un tweet au modèle
    # Cas 1 : le modèle n'est pas entraîné, une erreur est renvoyée
    # Cas 2 : le modèle est entraîné, le texte de tweet est préparé, la prédiction est réalisée avec en retour une variable binaire et les probabiltiés associées.

    def submit(self,sub):
        if not self.trained == True:
            raise Exception("Le modèle n'a pas été entraîné, vous devez lancer le script prepare.py")
        else:
            sub['text']=sub['text'].apply(clean)
            x=self.classifier.transform(sub['text'])
            proba = self.model.predict_proba(x)
            reponse = self.model.predict(x)
            return reponse,round((proba[0][0])*100),round((proba[0][1])*100)

