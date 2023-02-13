from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import *
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials = True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Tweet(BaseModel):
    var1: int
    var2: int

@app.get("/")
async def root():
    return {"message": "Bonjour, je suis l'API"}

@app.post("/check")
async def check(tweet: Tweet):
    var1 = tweet.var1
    var2 = tweet.var2
    model = ClassificationModel()
    model.train()
    X = np.array([[var1, var2]])
    y_pred = model.predict(X)
    return {"resultat": y_pred.tolist()}


@app.get("/predict")
def predict(var1: float,var2: float):
    y = var1 + var2
    return {"y": y}

@app.get("/test")
async def predict(var1: int, var2: int):
    model = ClassificationModel()
    model.train()
    X = np.array([[var1, var2]])
    y_pred = model.predict(X)
    return {"resultat": y_pred.tolist()}

#curl http://localhost:8000/predict?var1=2&var2=3


#curl -H "Content-Type: application/json" -X GET --data @data.json http://localhost:8000/predict
