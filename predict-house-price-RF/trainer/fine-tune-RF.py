import datetime, os, subprocess, sys, argparse, logging
import joblib

import pandas as pd
from xgboost import XGBRegressor

from google.cloud import storage

import hypertune
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir', default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
parser.add_argument("--dataset-url", dest="dataset_url",type=str)
parser.add_argument('--n_estimators', type=int)
parser.add_argument('--max_depth', type=int)
parser.add_argument('--min_samples_split', type=int)
parser.add_argument('--min_samples_leaf', type=int)

args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

def get_data():
    dataset_url = args.dataset_url
    logging.info("Getting training data from: {}".format(dataset_url))


    # Load data into pandas, then use `.values` to get NumPy arrays
    data = pd.read_csv(dataset_url)

    labels = data['target_col']
    features = data.drop('target_col',axis=1)

    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=0)

    return train_data, test_data, train_labels, test_labels


def train_model(train_data, train_labels):

    n_estimators = args.n_estimators
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf

    logging.info("Start training ...")

    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf)

    model.fit(train_data, train_labels)

    logging.info("Training completed")

    return model

def evaluate_model(model, test_data, test_labels):

    predictions = model.predict(test_data)
    # predictions = [round(value) for value in pred]
    # evaluate predictions
    r2 = r2_score(test_labels, predictions)
    logging.info(f"Evaluation completed with R2: {r2}")

    # report metric for hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='r2',
        metric_value=r2
    )
    return r2


train_data, test_data, train_labels, test_labels = get_data()
model = train_model(train_data, train_labels)
r2 = evaluate_model(model, test_data, test_labels)

artifact_filename = 'RF_model.pkl'

# Save model artifact to local filesystem (doesn't persist)
local_path = artifact_filename
joblib.dump(model, local_path)

# Upload model artifact to Cloud Storage

model_directory = os.environ['AIP_MODEL_DIR']
storage_path = os.path.join(model_directory, artifact_filename)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)


# save metric to local file system

metric_filename = "RF-metrics.json"

with open(metric_filename, "w") as f:
    f.write(f"{'r2: {r2}'}")

logging.info("Saving metrics to {}/RF-metrics.json". format(model_directory))

storage_path = os.path.join(model_directory, metric_filename)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(metric_filename)