# azure-demo



import azureml.core
from azureml.core import Workspace, Environment, Experiment, Datastore, Dataset, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute, DatabricksCompute
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.steps import HyperDriveStep, HyperDriveStepRun, PythonScriptStep, DatabricksStep
from azureml.pipeline.core import Pipeline, PipelineData, TrainingOutput
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, loguniform
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.core import Pipeline, StepSequence
