import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class TweetCheckerModel:

    def __init__(self):
        self.trained = False
        self.imported = False

    def load(self):
        if os.path.exists('model.gz') == False or os.path.exists('classifier.gz') == False:
            raise Exception("Aucune sauvegarde du modèle a été trouvé")
        else:
            self.model = joblib.load('model.gz')
            self.classifier = joblib.load('classifier.gz')
            self.trained = True

    def data(self,url):
        self.train = pd.read_csv(url, sep=',') # lecture du dataframe 
        self.imported = True


    def prepare(self):
        if not self.imported == True:
            raise Exception("La préparation est impossible car les données non pas été encore importées, vous devez lancer le script prepare.py")
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
            print("Le modèle a été entraîné et sauvegardé")
          

    def submit(self,sub):
        sub['text']=sub['text'].apply(clean)
        x=self.classifier.transform(sub['text'])
        proba = self.model.predict_proba(x)
        reponse = self.model.predict(x)
        return reponse,round((proba[0][0])*100),round((proba[0][1])*100)

