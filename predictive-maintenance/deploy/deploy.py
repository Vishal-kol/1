from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Load workspace
ws = Workspace.from_config(path="config.json")

# Register model
model = Model.register(workspace=ws,
                       model_path="predictive-maintenance/models/xgb_v1.pkl",
                       model_name="predictive-maintenance")

# Define environment
env = Environment.from_conda_specification(name="xgb-env", file_path="predictive-maintenance/requirements.txt")

# Define inference config
inference_config = InferenceConfig(entry_script="score.py", environment=env)

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
