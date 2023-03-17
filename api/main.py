from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import *
#uvicorn --reload main:app

# Création de l'objet FastAPI

app = FastAPI()

# On autorise tous les types de requête

app.add_middleware(
    CORSMiddleware,
    allow_credentials = True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Création de notre modèle vérification et chargement de la sauvegarde.
# Si les sauvegardes n'existent pas une erreur sera retournée, le script prepare.py devra être lancé pour entraîner le modèle avec un jeu de données

model = TweetCheckerModel()
model.load()

# Création du format de données, il est simple dans notre cas avec la récupération d'un seul champ texte

class Tweet(BaseModel):
    text: str

# Requête par défaut, elle retourne des informations simples concernant l'API

@app.get("/")
async def root():
    return {"message": "Bonjour, je suis l'API TweetChecker Version 1.0"}

# Requête pour la vérification des tweets : 
# Input : Fichier JSON contenant le contenu du tweet
# Ouput : Résultat de la prédiction du modèle

@app.post("/check")
async def check(tweet: Tweet):
    text = tweet.text
    df_tweet = pd.DataFrame({'id': [1], 'keyword': [''], 'location': [''], 'text': [text]})
    y_pred, prob1, prob2 = model.submit(df_tweet)
    return {"resultat": y_pred.tolist(),"truevalue":str(prob1),"falsevalue":str(prob2)}

