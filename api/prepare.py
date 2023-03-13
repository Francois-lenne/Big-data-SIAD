from model import *
# Lancement de la génération du modèle et son entraînement avec un jeu de données
model = TweetCheckerModel()
model.importData("https://raw.githubusercontent.com/Francois-lenne/Big-data-SIAD/main/data/train.csv")
model.prepare()
