import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from functions import *

class ClassificationModel:
    def __init__(self):
        self.model = LogisticRegression(solver='lbfgs')
        self.trained = False
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 1, 0, 1])

    def train(self):
        self.model.fit(self.X, self.y)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise Exception("Model not trained yet.")
        y_pred = self.model.predict(X)
        return y_pred




class TweetCheckerModel:

    def __init__(self):
        self.pipe = Pipeline([
            ("TdIdf",TfidfTransformer()),
            ("scaler", StandardScaler(with_mean=False)),

            ("classifier", LogisticRegression(random_state=0))
    
            ])
        self.param_grid = [
        
            {'classifier': [SVC(kernel='rbf', random_state=0, probability = True)],
            'classifier__C': [1,10]},
             ]
        self.grid = GridSearchCV(self.pipe, self.param_grid, verbose = 2, cv = 5)
        self.trained = False
        self.imported = False

    def importData(self,url):
        #url1 = 'https://raw.githubusercontent.com/Francois-lenne/Big-data-SIAD/main/train.csv' # le dataset est stocké dans un repo github afin d'avoir un lien dur sur la base
        self.train = pd.read_csv(url, sep=',') # lecture du dataframe 
        self.imported = True


    def prepare(self):
        if not self.imported == True:
            raise Exception("La préparation est impossible car les données non pas été encore importées")
        vectorizerCreate(self.train)
        self.X_train, self.y_train = DataPreparation(self.train,True)
        print("Entraînement OK")
       

        


    def train_model(self):
        self.grid.fit(self.X_train, self.y_train)
        joblib.dump(self.grid, 'model.gz')
        self.trained = True

    

    def submit(self,sub):
        if not self.trained == True and os.path.exists('model.gz') == False:
            raise Exception("Le model n'a pas été entraîné")
        self.model = joblib.load('model.gz')
        sub_trans = DataPreparation(sub,False)
        print(sub_trans)
        reponse = self.model.predict(sub_trans)
        return reponse

