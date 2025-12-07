# submit_job.py
from azure.ai.ml import MLClient, Input, command
from azure.identity import InteractiveBrowserCredential
import os

# 1. Authenticate (this will open a browser)
credential = InteractiveBrowserCredential()

# 2. Connect to your workspace
ml_client = MLClient(
    credential=credential,
    subscription_id="89ae6ad7-deb2-46c2-ba85-afbd87c8328b",  # Get from Azure Portal
    resource_group_name="CW2",
    workspace_name="CW2"
)

print(f"Connected to workspace: {ml_client.workspace_name}")

# 3. Define the job
job = command(
    code="./src",  # Folder with your training script
    command="python train_defect_model.py --data_path ${{inputs.data_path}}",
    inputs={
        "data_path": Input(
            type="uri_file",
            path="azureml:dataset:1"  # Your dataset name
        )
    },
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
    compute="CW2-Compute",  # Your compute cluster
    experiment_name="defect-prediction-cw2",
    display_name="defect-prediction-job"
)

# 4. Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Job details: {returned_job.services['Studio'].endpoint}")