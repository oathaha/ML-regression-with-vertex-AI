
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# The spec of the worker pools including machine type and Docker image
# Be sure to replace PROJECT_ID in the `image_uri` with your project.

worker_pool_specs = [{
    "machine_spec": {
        "machine_type": "n1-standard-4",
    },
    "replica_count": 1,
    "container_spec": {
        "image_uri": "gcr.io/house-price-prediction-439306/xgb-model:hypertune"
    }
}]


# Dictionary representing metrics to optimize.
# The dictionary key is the metric_id, which is reported by your training job,
# And the dictionary value is the optimization goal of the metric.

metric_spec={'r2':'maximize'}

# Dictionary representing parameters to optimize.
# The dictionary key is the parameter_id, which is passed into your training
# job as a command line argument,
# And the dictionary value is the parameter specification of the metric.

parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=0.001, max=1.0, scale="linear"),
    "subsample": hpt.DoubleParameterSpec(min=0.0, max=1, scale="linear"),
    "n_estimators": hpt.DiscreteParameterSpec(values=[10,20], scale=None),
    "max_depth": hpt.DiscreteParameterSpec(values=[2,3], scale=None)
}


# %%
my_custom_job = aiplatform.CustomJob(display_name='house-price-xgboost-job',
                              worker_pool_specs=worker_pool_specs,
                              staging_bucket='gs://cloud-ai-platform-07b92f9d-de58-4aea-81a9-1566224d6a62'
)

# %%
hp_job = aiplatform.HyperparameterTuningJob(
    display_name='house-price-xgboost-job',
    custom_job=my_custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=4,
    parallel_trial_count=2)


# %%
hp_job.run()

# %%



