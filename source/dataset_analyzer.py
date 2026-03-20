import os
import pandas as pd
import numpy as np


DATA_PATH = "datasets/raw_datasets"
OUTPUT_PATH = "datasets/meta_dataset.csv"


## Store meta features for all the datasets

def extract_meta_features(df):

    meta = {}

    meta["num_rows"] = df.shape[0]
    meta["num_columns"] = df.shape[1]
    meta["missing_ratio"] = df.isnull().sum().sum()/(df.shape[0]*df.shape[1])
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 0:
        meta["avg_std"] = numeric_df.std().mean()
        meta["avg_skewness"] = numeric_df.skew().mean() #skewness measures the asymmetry of the distribution of each numeric feature.
        meta["avg_variance"] = numeric_df.var().mean() # gives overall sense of how much variability exists across all numeric features.
        
        corr = numeric_df.corr().abs()
        meta["avg_correlation"] = corr.mean().mean()  #average correlation across all feature pairs, i.e., how redundant your features are.
    
    else:

        meta["avg_variance"] = 0
        meta["avg_skewness"] = 0
        meta["avg_correlation"] = 0
        meta["avg_std"] = 0

    meta["num_numeric_features"] = len(numeric_df.columns)
    meta["num_categorical_features"] = df.shape[1] - len(numeric_df.columns)

    return meta


def analyze_dataset(file_path):

   df = pd.read_csv(file_path)
   meta = extract_meta_features(df)
   
   meta["dataset_name"] = os.path.basename(file_path)
   
   return meta

# dataset files are stored in raw_datasets, scan this directory and analyze each dataset, then store the meta features in a csv file in meta_dataset
def scan_datasets():

    dataset_file =[]
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".csv"):
                dataset_file.append(os.path.join(root, file))
    
    return dataset_file


def main():
    dataset_files = scan_datasets()
    meta_features_list = []
    
    for file in dataset_files:
        
        try:

            meta = analyze_dataset(file)

            meta_features_list.append(meta)

            print(f"Analyzed: {file}")

        except Exception as e:

            print(f"Failed: {file} | {e}")

    df_meta = pd.DataFrame(meta_features_list)

    df_meta.to_csv(OUTPUT_PATH, index=False)

    print("\nMeta dataset created:", OUTPUT_PATH)
    
if __name__ == "__main__":
    main()

    

