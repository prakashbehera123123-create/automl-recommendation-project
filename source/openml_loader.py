import openml
import pandas as pd
import os

DOWNLOAD_PATH = "datasets/raw_datasets/openml"

DATASET_IDS = [
61,     # iris
37,     # diabetes
31,     # credit-g
1464,   # blood-transfusion
44,     # spambase
151,    # electricity
188,    # eucalyptus
300,    # isolet
1471,   # bank-marketing
42165,  # adult
41021,
40983,
40685,
40536,
40691,
41027,
41138,
40975,
40996,
41070
]

os.makedirs(DOWNLOAD_PATH, exist_ok=True)

for dataset_id in DATASET_IDS:

    try:

        dataset = openml.datasets.get_dataset(dataset_id)

        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

        df = X.copy()
        df["target"] = y

        filename = f"{dataset.name}.csv"

        path = os.path.join(DOWNLOAD_PATH, filename)

        df.to_csv(path, index=False)

        print(f"Downloaded {filename}")

    except Exception as e:

        print(f"Error downloading dataset {dataset_id}: {e}")