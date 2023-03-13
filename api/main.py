from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import *
#uvicorn --reload main:app


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials = True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Tweet(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Bonjour, je suis l'API"}

@app.post("/check")
async def check(tweet: Tweet):
    text = tweet.text
    model = TweetCheckerModel()
    df_tweet = pd.DataFrame({'id': [1], 'keyword': [''], 'location': [''], 'text': [text]})
    y_pred, prob1, prob2 = model.submit(df_tweet)
    return {"resultat": y_pred.tolist(),"truevalue":str(prob1),"falsevalue":str(prob2)}

#curl http://localhost:8000/predict?var1=2&var2=3


#curl -H "Content-Type: application/json" -X GET --data @data.json http://localhost:8000/predict
