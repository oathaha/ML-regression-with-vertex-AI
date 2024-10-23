FROM us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-6:latest

WORKDIR /

RUN pip install cloudml-hypertune

COPY trainer /trainer

ENTRYPOINT ["python", "-m", "trainer.fine-tune-xgboost", "--dataset-url", "gs://cloud-ai-platform-07b92f9d-de58-4aea-81a9-1566224d6a62/train_data.csv"]