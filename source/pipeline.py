
import os
import pandas as pd
from dataset_analyzer import extract_meta_features
from final_recommendation_engine import final_recommendation_engine

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

meta_df = pd.read_csv("datasets/meta_dataset.csv") # load the meta dataset which contains the meta features and best model for each dataset, this is needed for the similarity engine and meta learning model to work correctly
sample_df = pd.read_csv(file_path) ## sample dataset to extract meta features from

meta_dict = extract_meta_features(sample_df)  ## extract meta features from the sample dataset

new_meta_vector = [meta_dict[f] for f in FEATURE_ORDER] ## convert the meta features dictionary to a vector which can be used as input for the recommendation engine

def convert_to_dataframe(meta_dict): # convert the meta features dictionary to a dataframe with the same order of features as the meta dataset, this is needed for the similarity engine and meta learning model to work correctly
    return pd.DataFrame([meta_dict])[FEATURE_ORDER]

new_meta_df = pd.DataFrame([meta_dict])[FEATURE_ORDER]

result = final_recommendation_engine(new_meta_df, meta_df)  ## get final recommendations based on the meta features of the new dataset

print("Recommended model for the new dataset:", result)
