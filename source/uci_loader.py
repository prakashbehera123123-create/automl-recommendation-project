import pandas as pd
import os

from streamlit import columns

DOWNLOAD_PATH = "datasets/raw_datasets/uci"

DATASETS = {
"wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
"abalone": "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
"bank": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.csv",
"car": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
"breast_cancer": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data",
"heart_disease": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
"glass": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
"yeast": "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
"seeds": "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
"ecoli": "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
"ionosphere": "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
"sonar": "https://archive.ics.uci.edu/ml/machine-learning-databases/sonar/sonar.all-data",
"wine_quality": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
"student": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv",
"balance_scale": "https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
}

os.makedirs(DOWNLOAD_PATH, exist_ok=True)

for name, url in DATASETS.items():

    try:

        df = pd.read_csv(url, header= None)
        
        ## create feature names and target name
        df.columns = [f"feature_{i}" for i in range(df.shape[1]-1)] + ["target"]        

        path = os.path.join(DOWNLOAD_PATH, f"{name}.csv")
        

        df.to_csv(path, index=False)

        print(f"Downloaded {name}")

    except Exception as e:

        print(f"Error downloading {name}: {e}")