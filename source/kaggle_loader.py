import os
import subprocess
import zipfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()

DATASETS = [
"uciml/iris",
"uciml/wine-quality",
"uciml/breast-cancer-wisconsin-data",
"yasserh/housing-prices-dataset",
"rohanrao/air-quality-data-in-india",
"mlg-ulb/creditcardfraud",
"blastchar/telco-customer-churn",
"neuromusic/avocado-prices",
"nickj26/car-insurance-dataset",
"devansodariya/student-performance-data",
"yasserh/titanic-dataset",
"ruchi798/data-science-job-salaries",
"andonians/random-linear-regression",
"shubham47/online-retail",
"tejashvi14/movie-recommendation-system",
"shruthimech/amazon-product-reviews",
"shivamb/netflix-shows",
"rounakbanik/the-movies-dataset",
"tmdb/tmdb-movie-metadata",
"thedevastator/uncover-global-trends-in-health-data",
"fedesoriano/stroke-prediction-dataset",
"mathchi/diabetes-data-set",
"uciml/pima-indians-diabetes-database",
"kaushiksuresh147/customer-segmentation",
"jillanisofttech/customer-segmentation-dataset",
"uciml/red-wine-quality-cortez-et-al-2009",
"yasserh/student-marks-dataset",
"rajyellow46/wine-quality",
"imakash3011/customer-personality-analysis",
"thedevastator/loan-default-prediction"
]

DOWNLOAD_PATH = "datasets/raw_datasets/kaggle"

def download_dataset(dataset):

    print(f"\nDownloading: {dataset}")

    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    try:

        api.dataset_download_files(
            dataset,
            path=DOWNLOAD_PATH,
            unzip=True
        )

        print(f"{dataset} downloaded successfully")

    except Exception as e:

        print(f"Failed to download {dataset}: {e}")


def main():

    for dataset in DATASETS:

        download_dataset(dataset)


if __name__ == "__main__":

    main()