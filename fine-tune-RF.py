import datetime, os, subprocess, sys, argparse, logging, pickle

import pandas as pd

from google.cloud import storage

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir', default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
parser.add_argument("--dataset-url", dest="dataset_url",type=str)
parser.add_argument("--staging_bucket", type=str)


args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

def get_data():
    
    """
    This function retrieves the dataset URL from the `args` parameter, logs the
    retrieval process, reads the data using Pandas, and extracts the target column
    as labels. The features and labels are then split into training and validation sets
    with an 80-20 split.

    Returns:
    -----------
    tuple: A tuple containing four elements:
        - train_data (pd.DataFrame): The feature set for training.
        - test_data (pd.DataFrame): The feature set for testing.
        - train_labels (pd.Series): The labels for training.
        - test_labels (pd.Series): The labels for testing.
    """
    
    dataset_url = args.dataset_url
    logging.info("Getting training data from: {}".format(dataset_url))


    # Load data into pandas, then use `.values` to get NumPy arrays
    data = pd.read_csv(dataset_url)

    labels = data['target_col']
    features = data.drop('target_col',axis=1)

    # labels = labels.reshape((labels.size,))

    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=0)

    return train_data, test_data, train_labels, test_labels


def train_model(train_data, train_labels):
    """
    Trains a Random Forest Regressor model using the provided training data and labels.
    
    Parameters:
    -----------
    train_data : array-like, shape (n_samples, n_features)
        The input data to train the model. Each row corresponds to a sample, and each column corresponds to a feature.

    train_labels : array-like, shape (n_samples,)
        The target values for each sample in `train_data`.

    Returns:
    --------
    model : RandomForestRegressor
        A trained Random Forest Regressor model with the specified hyperparameters.
    """
    
    n_estimators = 5
    max_depth = 20
    min_samples_split = 3
    min_samples_leaf = 3

    logging.info("Start training ...")

    # Train RF model

    model = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf)

    model.fit(train_data, train_labels)

    logging.info("Training completed")

    return model



staging_bucket = args.staging_bucket


train_data, test_data, train_labels, test_labels = get_data()
model = train_model(train_data, train_labels)


## Save model artifact to local filesystem (doesn't persist)
artifact_filename = 'RF_model.pkl'
local_path = artifact_filename

with open(local_path, 'wb') as f:
    pickle.dump(model, f)

## Upload model artifact to Cloud Storage
model_directory = os.environ['AIP_MODEL_DIR']

logging.info("Saving metrics to {}/{}". format(model_directory, artifact_filename))

storage_path = os.path.join(model_directory, artifact_filename)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)
