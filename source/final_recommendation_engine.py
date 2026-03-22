
import joblib
import os
import pandas as pd
from source.similarity_engine import get_similar_datasets, similarity_recommendation


def load_artifact(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Train the model first.")
    return joblib.load(path)

def final_recommendation_engine(new_meta_df, meta_df):

    # -------------------------------
    # 1. Load model + scaler
    # -------------------------------
    meta_model = load_artifact("models/meta_model.pkl")
    scaler = load_artifact("models/scaler.pkl")

    # -------------------------------
    # 2. Validate features
    # -------------------------------
    if list(new_meta_df.columns) != list(scaler.feature_names_in_):
        raise ValueError("Feature mismatch between input and trained scaler")

    # -------------------------------
    # 3. Scale input
    # -------------------------------
    new_meta_scaled = scaler.transform(new_meta_df)

    # -------------------------------
    # 4. Meta-learning prediction
    # -------------------------------
    meta_probs = meta_model.predict_proba(new_meta_scaled)[0]
    meta_classes = meta_model.classes_

    # Top prediction
    meta_pred = meta_classes[meta_probs.argmax()]
    meta_conf = max(meta_probs)

    # -------------------------------
    # 5. Similarity-based prediction
    # -------------------------------
    similar_datasets, similarity_scores = get_similar_datasets(new_meta_df, meta_df)

    sim_result = similarity_recommendation(
        similar_datasets,
        similarity_scores,
        meta_pred
    )

    sim_pred = sim_result["similarity_prediction"]
    sim_conf = sim_result["similarity_confidence"]

    # -------------------------------
    # 6. HYBRID DECISION (IMPORTANT)
    scores = {}

    # Meta contribution
    scores[meta_pred] = 0.6 * float(meta_conf)

    # Similarity contribution
    scores[sim_pred] = scores.get(sim_pred, 0) + 0.4 * float(sim_conf)

    # Final decision
    final_algo = max(scores, key=scores.get)
    final_conf = scores[final_algo]

    # -------------------------------
    # 7. Output
    # -------------------------------
    return {
        "final_algorithm": final_algo,
        "confidence": round(float(final_conf), 3),
        "meta_prediction": meta_pred,
        "meta_confidence": round(float(meta_conf), 3),
        "similarity_prediction": sim_pred,
        "similarity_confidence": round(float(sim_conf), 3),
        "details": {
            k: float(v) for k, v in sim_result["all_similarity_scores"].items()
        }
    }
    
    