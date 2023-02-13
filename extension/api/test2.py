from model import *
import pandas as pd

tweet = "Je suis le tweet à tester, je suis là pour le modèle."

df_tweet = pd.DataFrame({'id': [1], 'keyword': [''], 'location': [''], 'text': [tweet]})

model_test = TweetCheckerModel()
print(model_test.submit(df_tweet))