import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI(title="ML Model API", description="Serving MLflow model with FastAPI")

# Load the best model from MLflow
RUN_ID = "3d06061842314165ab09da8c399419f1"  
model_uri = f"runs:/{RUN_ID}/best_random_forest_model"
model = mlflow.sklearn.load_model(model_uri)

# Request model for prediction input
class PredictionInput(BaseModel):
    instances: List[List[float]]  

# Define prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput):
    predictions = model.predict(data.instances).tolist()  
    return {"predictions": predictions}

# Root endpoint
@app.get("/")
def root():
    return {"message": "ML Model API is running!"}

