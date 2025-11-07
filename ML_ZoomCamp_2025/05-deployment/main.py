# main.py
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ----- 1. Créer l'application FastAPI -----
app = FastAPI(title="Lead Conversion Predictor API")

# ----- 2. Charger le modèle Pickle -----
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# ----- 3. Définir un modèle de données -----
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: float
    annual_income: float

# ----- 4. Créer une route d'accueil -----
@app.get("/")
def root():
    return {"message": "Welcome to the Lead Conversion Predictor API"}

# ----- 5. Créer la route de prédiction -----
@app.post("/predict")
def predict(lead: Lead):
    data = lead.dict()
    # Le modèle attend une liste de dictionnaires
    proba = model.predict_proba([data])[0, 1]
    return {"conversion_probability": round(float(proba), 3)}

# ----- 6. Lancer le serveur -----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9697)  # changer 9696 → 9697

