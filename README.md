# azure-demo
The general-deployet-pipeline file is the general jupyternotebook file to run the pipeline from end to end for your model deployment. Here the model deployment means to deploy an end-to-end ml pipeline to an online endpoint. You need to run it in an azureML workspace.

In your AML deployment pipelines, it is combined by multiple components based on your design. At this demo we will have 2 steps: the data cleasing step and a model trainning step. Each component is respresented by a python scrip. 

