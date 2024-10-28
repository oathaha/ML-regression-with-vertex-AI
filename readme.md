# ML-regression-with-vertex-AI

This repository demonstrates the example of how to build CI/CD pipeline on GitHub by using GitHub Action with Vertex AI in Google Cloud Platform.

To reproduce this pipeline, the first step is forking this repository and clone it to your own machine:

	git clone https://github.com/<your_repo_name>/ML-regression-with-vertex-AI.git

Then, follow the below steps to prepare Google Cloud Platform, GitHub Action, and script for running this pipeline.

### Google Cloud Platform preparation

Follow the steps below to prepare Google Cloud Platform for GitHub Action (you need to create new project if you don't have any project in Google Cloud Platform)

1. Data preparation: upload train.csv and test.csv in your bucket. You may need to create new bucket if there is no bucket in Google Cloud Platform
		
2.  Authentication setting: follow the steps below to set up authentication for GitHub

	1. Create service account with the following permissions
		-  Compute Admin
		-  Compute Storage Admin
		-  Environment and Storage Object Administrator
		-  Owner
		-  Storage Folder Admin
		-  Storage Object Admin
	2. Click at the newly created service account
	3. Click at KEYS -> ADD KEY -> Create new key
	4. Enter detail as seen on the screen
	5. Look at the json file that is downloaded when completing step 4.
	6. Open the JSON file in step 5. and replace all new line characters with space

### GitHub Action Preparation
Follow the steps below to prepare GitHub Action

1. Go to the forked repository
2. Click at Settings -> Secrets and variables -> Actions
3. Click "Add new repository secret" then paste the content in JSON from step 2.6.
4. Click "Add secret"

### Source code modfication

Follow the steps below to prepare source code to run with your Google Cloud Platform

1. Go to the forked repository
2. Open `.github/workflows/vertex_ai_deployment.yml`
3. At the line `name: authenticate gcloud`,  make changes at `project_id` and `service_account` to match your project. Also, change `FOR_GCP` to match your name of secret that is created in step 2.9.
4. At the line `name: fine-tune model`, make changes at the arguments `--project_id`, `--staging_bucket`, `--train_dataset_url`, `--test_dataset_no_label_url` and `test_dataset_with_label_url` to match your project and your file in your bucket
	
After the script files are commited and pushed, go to tab `Actions`  of the forked repository to view the newly created pipeline. You can also go to your Google Cloud Platform to see that a model is trained on VertexAI and is saved to the bucket in your project.
