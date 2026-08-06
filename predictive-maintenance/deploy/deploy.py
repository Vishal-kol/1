import os
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Load workspace
from azureml.core.authentication import ServicePrincipalAuthentication

print("Tenant ID:", os.getenv("AZURE_TENANT_ID"))


auth = ServicePrincipalAuthentication(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    service_principal_id=os.getenv("AZURE_CLIENT_ID"),
    service_principal_password=os.getenv("AZURE_CLIENT_SECRET")
    #authority=f"https://login.microsoftonline.com/{os.getenv('AZURE_TENANT_ID')}"
)
ws = Workspace(subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
               resource_group="AI-workspace",
               workspace_name="mlops",
               auth=auth)

# Register model
model = Model.register(workspace=ws,
                       model_path="predictive-maintenance/models/xgb_v1.pkl",
                       model_name="predictive-maintenance")

# Define environment
env = Environment.from_pip_requirements(
    name="xgb-env",
    file_path="predictive-maintenance/requirements.txt"
)

# Define inference config
inference_config = InferenceConfig(entry_script="predictive-maintenance/src/score.py", environment=env)

# Define deployment config
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy service
service = Model.deploy(workspace=ws,
                       name="predictive-maintenance-service",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)
print("Scoring URI:", service.scoring_uri)

#gmail