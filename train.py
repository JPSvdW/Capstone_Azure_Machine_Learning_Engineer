from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset
from azureml.data.dataset_factory import TabularDatasetFactory

breast_cancer_url = 'https://raw.githubusercontent.com/JPSvdW/Machine_Learning_Engineer_with_Microsoft_Azure_Nanodegree_Capstone_Project/main/data/breast-cancer.csv'

breast_cancer_dataset = TabularDatasetFactory.from_delimited_files(breast_cancer_url)

print(breast_cancer_dataset.take(3).to_pandas_dataframe())

def clean_data(data):

    train_x = data.to_pandas_dataframe().dropna()
    train_x["diagnosis"] = train_x.diagnosis.apply(lambda s: 1 if s == "M" else 0)
    train_x = train_x.rename(columns={"diagnosis" : "malignant"})
    train_x = train_x.drop(columns=["id"])

    train_y = train_x.pop("malignant")
    return train_x, train_y

x, y = clean_data(breast_cancer_dataset)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('output', exist_ok=True)
    joblib.dump(model, 'output/best-hyperdrive.pkl')

if __name__ == '__main__':
    main()
