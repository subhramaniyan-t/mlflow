import os
import tarfile
import pandas as pd
import numpy as np
from six.moves import urllib
import logging
import logging.config
import argparse 
from sklearn.model_selection import (
    train_test_split,
)
base_path = r"/home/subhramaniyan/mle-training/"

parser = argparse.ArgumentParser()
parser.add_argument("output_folder")

args = parser.parse_args()

print(args.output_folder)
input_dir = (os.getcwd())
help = """
ingest_data script

ingest_data  [output_folder_name]

"""
__all__ = ["get_data"]



logging.basicConfig(filename=base_path + "logs//" + 'ingest_data.log', encoding='utf-8', level=logging.DEBUG)

#"\\wsl.localhost\Ubuntu\home\subhramaniyan\mle-training\datasets\housing\housing.csv"
def load_data():
    
    Input_path = r"/home/subhramaniyan/mle-training/datasets/housing/housing.csv"
    #housing = pd.read_csv("housing.csv")
 
    housing = pd.read_csv(Input_path)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    print(housing.columns)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    train_set.to_csv(base_path +  args.output_folder + "train.csv")
    test_set.to_csv(base_path +  args.output_folder + "test.csv")
    logging.debug("Ingestion done")

load_data()
