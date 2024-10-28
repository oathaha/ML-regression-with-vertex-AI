from google.cloud import aiplatform

from sklearn.metrics import r2_score
import pandas as pd
import random, string, json, logging, argparse

# Setup argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--project_id', type=str, required=True, help="GCP project ID")
parser.add_argument('--staging_bucket', type=str, required=True, help="GCS bucket for staging artifacts")
parser.add_argument("--train_dataset_url", type=str, required=True, help="URL for training dataset")
parser.add_argument('--test_dataset_url', type=str, required=True, help="URL for test dataset")
args = parser.parse_args()

# Set logging level to INFO
logging.getLogger().setLevel(logging.INFO)


# Function to generate a unique identifier
def generate_uuid(length: int = 8) -> str:
    """Generates a random UUID with specified length."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

# Generate UUID for this model deployment session
UUID = generate_uuid()

# Define GCS bucket paths
project_id = args.project_id
staging_bucket = args.staging_bucket
MODEL_DIR = f"{staging_bucket}/{UUID}"

# Define container images for training and testing
train_docker_img = 'us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest'
test_docker_img = 'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'

# Initialize Vertex AI platform with specified project and staging bucket
aiplatform.init(project=project_id, staging_bucket=staging_bucket)

# Create a Custom Training Job using Vertex AI
job = aiplatform.CustomTrainingJob(
    display_name='RF for deployment',
    script_path='./fine-tune-RF.py',
    container_uri=train_docker_img,
)

logging.info('Training model...')

# Run training job with the dataset URL as an argument
job.run(base_output_dir=MODEL_DIR, sync=True, args=[f"--dataset-url={args.train_dataset_url}", f"--staging_bucket={staging_bucket}"])

logging.info('Training completed')


# Define model path for deployment
MODEL_DIR += "/model"
model_path_to_deploy = MODEL_DIR


# Upload the trained model to Vertex AI
model = aiplatform.Model.upload(
    display_name='house-price-prediction-for-deployment',
    artifact_uri=MODEL_DIR,
    serving_container_image_uri=test_docker_img,
    sync=True
)

model.wait()
logging.info('Model upload completed')


# Set parameters for batch prediction
MIN_NODES = 1
MAX_NODES = 1

# Run batch prediction job on test data
batch_predict_job = model.batch_predict(
    job_display_name='rf-batch-prediction',
    gcs_source=args.test_dataset_url,
    gcs_destination_prefix=staging_bucket,
    instances_format="jsonl",
    predictions_format="jsonl",
    model_parameters=None,
    machine_type="n1-standard-2",
    starting_replica_count=MIN_NODES,
    max_replica_count=MAX_NODES,
    sync=True,
)

logging.info("Running batch prediction job")

batch_predict_job.wait()

logging.info("Batch prediction completed")


# Retrieve and parse batch prediction results
bp_iter_outputs = batch_predict_job.iter_outputs()
prediction_results = []

for blob in bp_iter_outputs:
    if blob.name.split("/")[-1].startswith("prediction.results"):
        prediction_results.append(blob.name)


# Process prediction results into a list
results = []
for prediction_result in prediction_results:
    gfile_name = f"gs://{bp_iter_outputs.bucket.name}/{prediction_result}"
    content = blob.download_as_string().decode().split('\n')
    
    for line in content:
        if line.strip():
            results.append(json.loads(line)['prediction'])


# Load test dataset and calculate R2 score for model evaluation
test_df = pd.read_csv(args.test_dataset_url)
labels = test_df['target_col'].tolist()

r2 = r2_score(labels, results)
logging.info(f'R2 score = {r2:.2f}')


# Deploy the model if it meets the performance criteria
if r2 >= 0.9:
    DEPLOYED_NAME = 'house-price-prediction'
    DEPLOY_COMPUTE = "n1-standard-4"
    TRAFFIC_SPLIT = {"0": 100}

    endpoint = model.deploy(
        deployed_model_display_name=DEPLOYED_NAME,
        traffic_split=TRAFFIC_SPLIT,
        machine_type=DEPLOY_COMPUTE,
        min_replica_count=MIN_NODES,
        max_replica_count=MAX_NODES,
    )
    endpoint.wait()
    logging.info('Model deployment completed')
