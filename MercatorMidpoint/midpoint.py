#!/usr/bin/env python

from flask import Flask, request, make_response, jsonify, Response
import os

from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + "/inbound-bee-381420-3b5ab19a2a50.json"

app = Flask(__name__)

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
def get_image():
    data = request.get_json()

    response = predict_custom_trained_model_sample(
        project="inbound-bee-381420",
        endpoint_id="6563798544799498240",
        location="us-central1",
        instances={ "world": data['world'], "coords": data['coords'], "vectors": data['vectors'] }
    )

    return jsonify({'images': response.predictions[0], 'vectors': response.predictions[1]})

@app.route('/')
def home():
    return ""

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5555, debug=True)
    except Exception as ex:
        print(ex, file=stderr)
        exit(1)
