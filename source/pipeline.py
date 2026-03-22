
import os
import pandas as pd
from source.dataset_analyzer import extract_meta_features
from source.final_recommendation_engine import final_recommendation_engine

file_path = "datasets/sample_data/milk_quality_data.csv"

FEATURE_ORDER = [
    "num_rows",
    "num_columns",
    "missing_ratio",
    "avg_std",
    "avg_skewness",
    "avg_variance",
    "avg_correlation",
    "num_numeric_features",
    "num_categorical_features"
]
def run_recommendation(df: pd.DataFrame):

    # Step 1: Extract meta features
    meta_dict = extract_meta_features(df)

    # Step 2: Convert to DataFrame (ordered)
    new_meta_df = pd.DataFrame([meta_dict])[FEATURE_ORDER]

    # Step 3: Load meta dataset
    meta_df = pd.read_csv("datasets/meta_dataset.csv")

    # Step 4: Get recommendation
    result = final_recommendation_engine(new_meta_df, meta_df)

    return result


