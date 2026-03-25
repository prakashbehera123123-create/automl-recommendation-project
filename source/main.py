from fastapi import FastAPI, UploadFile, File
import pandas as pd
from source.pipeline import run_recommendation
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


## input file → DataFrame function


def load_file_to_dataframe(file: UploadFile):
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(file.file)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(file.file)

        elif filename.endswith(".json"):
            return pd.read_json(file.file)

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
    
    
    # handle the file processing and conversion to DataFrame

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Step 1: Convert file → DataFrame
    df = load_file_to_dataframe(file)

    # Step 2: Run pipeline
    result = run_recommendation(df)

    return result