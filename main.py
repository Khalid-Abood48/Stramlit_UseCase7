from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the K-Means model
model = joblib.load('Data/final_data.csv')

class PlayerFeatures(BaseModel):
    height: float
    age: float
    appearance: float
    goals: float
    assists: float
    current_value: float
    position_encoded: int

@app.post("/predict")
def predict(player: PlayerFeatures):
    features = [[
        player.height,
        player.age,
        player.appearance,
        player.goals,
        player.assists,
        player.current_value,
        player.position_encoded
    ]]
    prediction = model.predict(features)
    return {"prediction": prediction[0]}

