
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def get_similar_datasets(new_meta, meta_df, top_k=3):

    # Remove non-feature columns
    drop_cols = [col for col in ["dataset_name", "best_model"] if col in meta_df.columns]
    feature_df = meta_df.drop(columns=drop_cols)

    # Compute similarity
    similarities = cosine_similarity(new_meta, feature_df)[0]

    # Get top K similar datasets
    top_indices = similarities.argsort()[-top_k:][::-1]

    similar_rows = meta_df.iloc[top_indices]
    similarity_scores = similarities[top_indices]

    return similar_rows, similarity_scores




def similarity_recommendation(similar_rows, similarity_scores, meta_pred):

    # Get algorithms from similar datasets
    algos = similar_rows["best_model"]

    # Count frequency
    algo_counts = Counter(algos)

    # Most common from similarity
    sim_pred = algo_counts.most_common(1)[0][0]

    # Average similarity score (confidence)
    sim_conf = similarity_scores.mean()

    return {
        "similarity_prediction": sim_pred,
        "similarity_confidence": sim_conf,
        "meta_prediction": meta_pred
    }