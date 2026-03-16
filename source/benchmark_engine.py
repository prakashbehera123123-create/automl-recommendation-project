import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Dataset path

DATA_PATH = "datasets/raw_datasets"
META_PATH = "datasets/meta_dataset.csv"


# function for selecting the model for regression or classification

def get_models(problem_type):
    
    if problem_type == "classification":
        models ={
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
            "GradientBoosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier()
        }
    else:
        
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "KNNRegressor": KNeighborsRegressor(),
            "SVR": SVR(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "XGBoostRegressor": XGBRegressor()
        }
    return models

def find_best_model(X_train, X_test, y_train, y_test, problem_type):

    models = get_models(problem_type)

    best_model = None
    best_score = -999

    for name, model in models.items():

        try:

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)

            if score > best_score:
                best_score = score
                best_model = name

        except:
            continue

    return best_model


# data loading and processing

def process_dataset(dataset_path):
    
    
    try:

        df = pd.read_csv(dataset_path)
    
        df = df.dropna()
        
        if df.shape[0] > 100000:
            df = df.sample(100000, random_state=42)

        target_column = df.columns[-1]
        if df[target_column].nunique() < 2:
            return None, None

        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Drop columns with too many unique values
        for col in X.columns:

            if X[col].dtype == "object":

                if X[col].nunique() > 50:
                    X = X.drop(columns=[col])
        
        X = pd.get_dummies(X, drop_first=True)
        
        if X.shape[1] == 0:
            print("Dataset", dataset_path, "has no usable features")
            return None
        
        if y.dtype == "object":
                problem_type = "classification"
        else:
                problem_type = "regression"
                
                
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
                
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        
        
        best_model = find_best_model(X_train, X_test, y_train, y_test, problem_type)

        dataset_name = os.path.basename(dataset_path)
        print("Processing dataset:", dataset_name)
        

        return dataset_name, best_model
        
    
    except Exception as e:

        print(f"Error processing dataset {dataset_path}: {e}")
        return None, None


# data_scanner

def scan_dataset():
    dataset_files =[]
    
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".csv"):
                dataset_files.append(os.path.join(root, file))
                
    return dataset_files

# Parallel processing of datasets

def main():
    
    datasets = scan_dataset()[:5]  # Limit to first 5 datasets for testing
    
    print("datasets found:", len(datasets))
    
    result = Parallel(n_jobs= -1)(delayed(process_dataset)(dataset) for dataset in datasets)
    
    results = [r for r in result if r[0] is not None]
    
    meta_df = pd.read_csv(META_PATH)
    
    for dataset_name, best_model in results:
        meta_df.loc[meta_df["dataset_name"] == dataset_name, "best_model"] = str(best_model)

        meta_df.to_csv(META_PATH, index=False)

        
        print(f"\nUpdated meta dataset for {dataset_name} with best model: {best_model}")
        
        
        print("\nBenchmarking complete")
    

if __name__ == "__main__":

    main() 

    
