import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'azureml-fsspec'])
from azureml.core import Run
import argparse
import os
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, ScriptRunConfig
from azureml.core.datastore import Datastore
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.fsspec import AzureMachineLearningFileSystem
import pandas as pd


def clean_data(base_data):
    preped_data = base_data.copy()
    
    preped_data["thal"] = preped_data["thal"].astype("category").cat.codes

    return preped_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path_train', type=str, dest='input_data_path_train')
    parser.add_argument('--input_data_path_predict', type=str, dest='input_data_path_predict')
    parser.add_argument('--processed_data_train', type=str, dest='processed_data_train')
    parser.add_argument('--processed_data_inputdata', type=str, dest='processed_data_inputdata')

    args = parser.parse_args()

    print(f'args={args}\n')
    print(f'input arg: {args.input_data_path_train}')
    # input_path= os.path.join(args.input_data_path, '')
    # print(f'input_path: {input_path}')

    fs = AzureMachineLearningFileSystem(args.input_data_path_train)
    # append csv files in folder to a list
    dflist = []
    for path in fs.ls():
        with fs.open(path) as f:
            dflist.append(pd.read_csv(f))

    # concatenate data frames
    df = pd.concat(dflist)
    df.head()

    fs2 = AzureMachineLearningFileSystem(args.input_data_path_predict)
    # append csv files in folder to a list
    dflist2 = []
    for path in fs2.ls():
        with fs2.open(path) as f:
            dflist2.append(pd.read_csv(f))

    # concatenate data frames
    df2 = pd.concat(dflist2)
    df2.head()

# cleaned data for prediction
    output_data2= clean_data(df2)
    output_path2 = os.path.join(args.processed_data_inputdata, 'output.csv')
    print(f'Output path: [{output_path2}]')
    output_data2.to_csv(output_path2)

    # run = Run.get_context()
    # ds__raw = run.input_datasets['input_path']
    # pdf_raw = ds_raw.to_pandas_dataframe()

    # cleaned data for trainning
    output_data = clean_data(df)
    output_path = os.path.join(args.processed_data_train, 'output.csv')
    print(f'Output path: [{output_path}]')
    output_data.to_csv(output_path)

    

print('*************************** End ***************************')