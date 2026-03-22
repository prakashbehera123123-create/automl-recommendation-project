# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import Counter
# import numpy as np


# def get_similar_datasets(new_meta_df, meta_df, top_k=3):

#     # Remove non-feature columns
#     drop_cols = [col for col in ["dataset_name", "best_model"] if col in meta_df.columns]
#     feature_df = meta_df.drop(columns=drop_cols)

#     #  Ensure same column order for similarity calculation
#     feature_df = feature_df[new_meta_df.columns]

#     # Safety check if columns match
#     if list(new_meta_df.columns) != list(feature_df.columns):
#         raise ValueError("Feature mismatch between new data and meta dataset")

#     # Compute similarity
#     similarities = cosine_similarity(new_meta_df, feature_df)[0]

#     # Get top K similar datasets
#     top_indices = similarities.argsort()[-top_k:][::-1]

#     similar_rows = meta_df.iloc[top_indices]
#     similarity_scores = similarities[top_indices]

#     return similar_rows, similarity_scores


#    # (Weighted Recommendation)
# def similarity_recommendation(similar_rows, similarity_scores, meta_pred):

#     algos = similar_rows["best_model"].values

#     # Weighted voting instead of simple count
#     weighted_votes = {}

#     for algo, score in zip(algos, similarity_scores):
#         weighted_votes[algo] = weighted_votes.get(algo, 0) + score

#     # Best algorithm based on weighted similarity
#     sim_pred = max(weighted_votes, key=weighted_votes.get)

#     # Confidence (normalized)
#     sim_conf = np.mean(similarity_scores)

#     return {
#         "similarity_prediction": sim_pred,
#         "similarity_confidence": float(round(sim_conf, 3)),
#         "meta_prediction": meta_pred,
#         "all_similarity_scores": dict(zip(algos, similarity_scores))
#     }