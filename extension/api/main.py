from fastapi import FastAPI, Request, Response
from model import *
import json

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Bonjour, je suis l'API"}

@app.post("/predictjson")
async def predict(request: Request):
    data = await request.json()
    var1 = data.get("var1")
    var2 = data.get("var2")
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
