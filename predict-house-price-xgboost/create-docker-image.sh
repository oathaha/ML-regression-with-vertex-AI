PROJECT_ID='house-price-prediction-439306'
IMAGE_URI="gcr.io/$PROJECT_ID/xgb-model:deployment"

gcloud auth configure-docker
docker build ./ -t $IMAGE_URI
docker push $IMAGE_URI