from xml.parsers.expat import model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


# Dataset path

meta_data_path = "datasets/meta_dataset.csv"

def meta_learning_model(meta_dataset_path):

    meta_df = pd.read_csv(meta_dataset_path)
    
    meta_df = meta_df.dropna(subset=["best_model"])

    drop_cols = [col for col in ["dataset_name", "best_model"] if col in meta_df.columns]
    X = meta_df.drop(columns=drop_cols)
    y = meta_df["best_model"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    

    print("classification report: ", classification_report(y_test, y_pred))
    print(f"\nMeta-learning model accuracy: {accuracy:.4f}")
    
    #save model and scaler
    
    joblib.dump(model, "meta_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    
    return model, scaler

meta_learning_model("datasets/meta_dataset.csv")



