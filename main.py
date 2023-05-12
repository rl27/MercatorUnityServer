#!/usr/bin/env python

from flask import Flask, request, jsonify, Response
from image_sampler.ImageSampler import ImageSampler
import json
from io import BytesIO
import base64

#-----------------------------------------------------------------------

app = Flask(__name__)
sampler = ImageSampler()
print("app and sampler created")

#-----------------------------------------------------------------------

using_midpoint = True

@app.route('/get_image', methods=['POST'])
def get_image():
    if using_midpoint:
        data = request.get_json()['instances'][0]
    else:
        data = request.get_json()  # Data format for communicating directly with Unity

    world_data = data['world']
    set_of_coords = data['coords']
    latent_vectors = data['vectors']
    sentence = data['sentence']

    sigma = data.get('sigma')
    lengthscale = data.get('lengthscale')
    
    ims = []
    vectors = []
    if sentence != "":
        print(sentence)
        
        world_data.append(set_of_coords[0])
        set_of_coords.pop(0)
        im, vector = sampler.generate_image_from_sentence(sentence)
        latent_vectors.append(vector)

        ims = [im]
        vectors = [vector]

    new_ims, new_latent_vectors = sampler.generate_images_for_megatile(world_data, set_of_coords, latent_vectors, sigma, lengthscale)

    ims = ims + new_ims
    vectors = vectors + new_latent_vectors

    # Convert PIL images to byte arrays, then to strings; place them all in one string, delimited by spaces
    images = ims_to_string(ims)
    
    if using_midpoint:
        return jsonify({ 'predictions': { 'images': images, 'vectors': str(vectors)} })
    else:
        return jsonify({ 'images': images, 'vectors': str(vectors) })  # Data format for communicating directly with Unity

# Health check route
@app.route("/isalive")
def isalive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

# https://stackoverflow.com/a/41444786
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
        app.run(host='0.0.0.0', port=8080, debug=True)
    except Exception as ex:
        print(ex, file=stderr)
        exit(1)