import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse 
import logging 

__all__ = ["get_score"]
base_path = r"/home/subhramaniyan/mle-training/"


parser = argparse.ArgumentParser()
parser.add_argument("model_folder")
parser.add_argument("dataset_folder")
args = parser.parse_args()


logging.basicConfig(filename=base_path + "logs//" + 'score.log', encoding='utf-8', level=logging.DEBUG)

def get_score():
    final_model = pickle.load(open(base_path  + args.model_folder + "model.pkl", "rb"))
    imputer = pickle.load(open(base_path  + args.model_folder + "imputer.pkl", "rb"))
    test_set = pd.read_csv(base_path  + args.dataset_folder + "test.csv")
    X_test = test_set.drop("median_house_value", axis=1)
    y_test = test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print(final_rmse)
    logging.debug("Scoring done")
get_score()

