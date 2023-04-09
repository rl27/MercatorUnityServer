# FROM python:3.9-slim
# FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
FROM us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-12:latest
# https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers#pytorch

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Launch server
EXPOSE 8080
ENTRYPOINT ["python", "main.py"]

# CMD ["gunicorn", "main:app", "--timeout=0", "--preload", "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]