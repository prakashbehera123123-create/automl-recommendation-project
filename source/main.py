from fastapi import FastAPI, UploadFile, File
import pandas as pd
from source.pipeline import run_recommendation

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    result = run_recommendation(df)

    return result