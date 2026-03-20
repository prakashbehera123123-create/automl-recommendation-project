import joblib
import pandas as pd
from similarity_engine import get_similar_datasets, similarity_recommendation

def final_recommendation_engine(new_meta, meta_df):
    # Load meta-learning model and scaler
    meta_model = joblib.load("meta_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    #new input scale
    new_meta_scaled = scaler.transform(new_meta)

    # Get similar datasets and their similarity scores
    similar_datasets, similarity_scores = get_similar_datasets(new_meta, meta_df)

    # Predict best model using meta-learning model
    new_meta_scaled =scaler.transform(new_meta) 
    predicted_model = meta_model.predict(new_meta_scaled)[0]

    # Get recommendations based on similarity and predicted model
    recommendations = similarity_recommendation(similar_datasets, similarity_scores, predicted_model)

    return recommendations

