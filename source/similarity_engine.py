import numpy as np
import pandas as pd
import faiss
from collections import Counter


def get_similar_datasets(new_meta_df, meta_df, top_k=3):

    # Remove non-feature columns
    drop_cols = [col for col in ["dataset_name", "best_model"] if col in meta_df.columns]
    feature_df = meta_df.drop(columns=drop_cols)

    # Ensure same column order
    feature_df = feature_df[new_meta_df.columns]

    # Convert to numpy float32
    X = np.ascontiguousarray(feature_df.values.astype("float32"))
    query = np.ascontiguousarray(new_meta_df.values.astype("float32"))
    # Normalize for cosine similarity
    faiss.normalize_L2(X)
    faiss.normalize_L2(query)

    # Build FAISS index (cosine via inner product)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # Search
    similarity_scores, indices = index.search(query, top_k)

    top_indices = indices[0]
    scores = similarity_scores[0]

    similar_rows = meta_df.iloc[top_indices]

    return similar_rows, scores


# Weighted Recommendation (same as before)
def similarity_recommendation(similar_rows, similarity_scores, meta_pred):

    algos = similar_rows["best_model"].values

    weighted_votes = {}

    for algo, score in zip(algos, similarity_scores):
        weighted_votes[algo] = weighted_votes.get(algo, 0) + float(score)

    sim_pred = max(weighted_votes, key=weighted_votes.get)

    sim_conf = float(np.mean(similarity_scores))

    return {
        "similarity_prediction": sim_pred,
        "similarity_confidence": round(sim_conf, 3),
        "meta_prediction": meta_pred,
        "all_similarity_scores": {
            algo: float(score) for algo, score in zip(algos, similarity_scores)
        }
    }