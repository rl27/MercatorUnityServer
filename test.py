#!/usr/bin/env python

from flask import Flask, request, jsonify, Response
from image_sampler.ImageSampler import ImageSampler
import json
from io import BytesIO
import base64
import numpy as np
import time

#-----------------------------------------------------------------------

app = Flask(__name__)
sampler = ImageSampler()
print("app and sampler created")

#-----------------------------------------------------------------------

@app.route('/get_image', methods=['POST'])
def get_image():
    # data = request.get_json()['instances'][0]
    data = request.get_json()  # Data format for communicating directly with Unity
    world_data = data['world']
    set_of_coords = data['coords']
    latent_vectors = data['vectors']
    sentence = data['sentence']

    sigma = data.get('sigma')
    lengthscale = data.get('lengthscale')
    
    target = None
    vectors = []
    if sentence != "":
        
        target = np.random.normal(0, 1, 512)
        target = target / np.linalg.norm(target)
        target = target.tolist()
        
        '''
        world_data.append(set_of_coords[0])
        set_of_coords.pop(0)
        _, vector = sampler.generate_image_from_sentence(sentence)
        latent_vectors.append(vector)
        vectors = [vector]

        target = vector + np.random.normal(0, 0.05, 512)
        target = target / np.linalg.norm(target)
        target = target.tolist()
        '''

    # ims, vectors = sampler.generate_images_for_megatile(world_data, set_of_coords, latent_vectors, sigma, lengthscale)
    # images = ims_to_string(ims)

    vectors = sampler.sample_latent_vector(world_data, set_of_coords, latent_vectors, sigma, lengthscale)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors.tolist()
    images = ""
    time.sleep(0.3)

    return jsonify({ 'images': images, 'vectors': str(vectors), 'target': str(target)[1:-1] })  # Data format for communicating directly with Unity

    # targeted init 
    '''
    target = None
    vectors = []
    if sentence != "":
        
        # target = np.random.normal(0, 1, 512)
        # target = target / np.linalg.norm(target)
        # target = target.tolist()
        
        world_data.append(set_of_coords[0])
        set_of_coords.pop(0)
        _, vector = sampler.generate_image_from_sentence(sentence)
        latent_vectors.append(vector)
        vectors = [vector]

        target = vector + np.random.normal(0, 0.05, 512)
        target = target / np.linalg.norm(target)
        target = target.tolist()

    # ims, vectors = sampler.generate_images_for_megatile(world_data, set_of_coords, latent_vectors, sigma, lengthscale)
    # images = ims_to_string(ims)

    vectors = vectors + sampler.sample_latent_vector(world_data, set_of_coords, latent_vectors, sigma, lengthscale)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors.tolist()
    images = ""
    time.sleep(0.3)
    '''

def convertToPNG(im):
    with BytesIO() as f:
        im.save(f, format='PNG') # convert the PIL image to byte array
        return f.getvalue()

def ims_to_string(ims):
    images = str(base64.b64encode(convertToPNG(ims[0])))[2:-1]
    for i in range(1, len(ims)):
        images = '{} {}'.format(images, str(base64.b64encode(convertToPNG(ims[i])))[2:-1])
    return images

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5555, debug=True)
    except Exception as ex:
        print(ex, file=stderr)
        exit(1)