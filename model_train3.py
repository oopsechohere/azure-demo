import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'xgboost'])

import azureml.core
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, Run

import os
import shutil
import urllib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import argparse
import os
import glob
import shutil
import joblib
import numpy as np
import pandas as pd
from pandas import read_csv


import mlflow
import mlflow.sklearn

from sklearn import __version__ as sklearnver
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from azureml.core.model import Model

def split_data(preped_data):
    preped_data = preped_data.copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
    preped_data.drop("target", axis=1), preped_data["target"], test_size=0.2
        )

    return preped_data, X_train,X_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned_data_train', type=str, dest='cleaned_data_train')
    parser.add_argument('--cleaned_data_predict', type=str, dest='cleaned_data_predict')
    # parser.add_argument('--output_modelpath', type=str, dest='output_modelpath')
    # parser.add_argument('--max_depth', type=int, dest='max_depth')

    args = parser.parse_args()
    print(f'args={args}')

    run = Run.get_context()
    print(f'run.input_datasets={run.input_datasets}')
    
    ds_raw = run.input_datasets['cleaned_data_train']
    pdf_raw = ds_raw.to_pandas_dataframe()

    print(f'pdf_raw.head()')
    print(pdf_raw.head())

    # pdf_raw['target'] = pdf_raw['target'].astype(int)
    
    pdf_raw = pdf_raw.apply(LabelEncoder().fit_transform)
    # preped_data, X_train,X_test, y_train, y_test = split_data(pdf_raw)
    X_train, X_test, y_train, y_test = train_test_split(
    pdf_raw.drop("target", axis=1), pdf_raw["target"], test_size=0.1
    )

    print(f'X_train.head() = {X_train.head()}')
    print(f'y_test.head() = {y_test.head()}')
    
    # get the input data for prediction:
    ds_raw_predict = run.input_datasets['cleaned_data_predict']
    pdf_raw_predict = ds_raw_predict.to_pandas_dataframe()
    print(f'pdf_raw_predict.head()')
    print(pdf_raw_predict.head())
    pdf_raw_predict = pdf_raw_predict.apply(LabelEncoder().fit_transform)


    ws = run.experiment.workspace

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        model = XGBClassifier(eval_metric="logloss")
        print(X_train.head())
        print(y_train.head())
        # log_reg = LogisticRegression(solver=args.solver, max_iter=args.max_iter, penalty=args.penalty, tol=args.tol)
        fitted_model = model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        isdir = os.path.isdir("outputs")
        if isdir:
            shutil.rmtree("outputs")
        mlflow.sklearn.save_model(fitted_model, "outputs")
        # run = Run.get_context()
        # run.upload_file('./outputs/model.pkl', args.output_modelpath)
        model = Model.register(model_name='demo1_sdk_model', model_path='./outputs/model.pkl', workspace = ws)

        # #make prediction  
        # fitted_model.predict()
        prediction_output = fitted_model.predict(pdf_raw_predict)
        print(f'prediction output : {prediction_output}')
