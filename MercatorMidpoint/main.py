#!/usr/bin/env python

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os

from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# Use Google service account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + "/inbound-bee-381420-3b5ab19a2a50.json"

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Adapted from: https://github.com/googleapis/python-aiplatform/blob/main/samples/snippets/prediction_service/predict_custom_trained_model_sample.py
def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    instances = instances if type(instances) == list else [instances]
    instances = [ json_format.ParseDict(instance_dict, Value()) for instance_dict in instances ]

    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())

    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    return client.predict(endpoint=endpoint, instances=instances, parameters=parameters)


@app.route('/get_image', methods=['POST'])
@cross_origin()
def get_image():
    data = request.get_json()

    response = predict_custom_trained_model_sample(
        project="inbound-bee-381420",
        endpoint_id="52684199156514816",
        location="us-central1",
        instances=data
    )

    return jsonify({'images': response.predictions[0], 'vectors': response.predictions[1]})


@app.route('/')
def home():
    return ""
