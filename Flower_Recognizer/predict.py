import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings("ignore")

import numpy as np

np.seterr(all="ignore")

import pandas as pd

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.losses.sparse_softmax_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy


import tensorflow_hub as hub

hub.KerasLayer._original_keras_layer_class = None

from PIL import Image
import argparse
import json
import tf_keras
from flask import Flask, request, jsonify


my_image_path = './test_images/hard-leaved_pocket_orchid.jpg'
my_model = tf_keras.models.load_model("Flower_Recognizer.h5", custom_objects={'KerasLayer': hub.KerasLayer})


def process_image(image):
    pre_image_size = 224

    image = tf.convert_to_tensor(image, dtype=tf.float32) # Convert the datatype to float
    image = tf.image.resize(image, (pre_image_size, pre_image_size)) # Convert the image to (224, 224, 3)
    image /= 255 # rescale the image from scope 0-255 -> 0-1
    image = image.numpy() # re-convert it to numpy array
    return image

def predict(path_image= my_image_path, model=my_model, names_path= "label_map.json", k=1):
    image = Image.open(path_image) # open the image
    image = np.asarray(image) # convert it to numpy
    image = process_image(image) # pre-process the image
    image = np.expand_dims(image, axis=0) # convert (224,224,3) to (1,224,224,3)

    with open(f"{names_path}", "r") as file: # open the .json file
        name_classes = json.load(file)

    prediction_dict = model.predict(image) # get the predection

    # for me, It's easier to deal with pandas
    pandas_predictions = pd.DataFrame(prediction_dict)
    sorted_prediction = pandas_predictions.T.sort_values(by=0, ascending=False).head(k).T
    probs = sorted_prediction.values[0]
    classes = sorted_prediction.keys().tolist()

    new_name_classes = []
    for cls in classes:
        new_name_classes.append(name_classes[f"{cls}"])

    return list(probs), list(new_name_classes)

# Got it from the first project
def get_input_args():
    parser = argparse.ArgumentParser(description="Add the four arguments")

    parser.add_argument('--image_path', type=str, default='./test_images/hard-leaved_pocket_orchid.jpg', help='path to the images')
    parser.add_argument('--model_path', type=str, default='Flower_Recognizer.h5',help='path to the model')
    parser.add_argument('--top_k', type=int, default=1, help='number of returned classes along side with their probs')
    parser.add_argument('--category_names', type=str, default='label_map.json', help='category_names')

    return parser.parse_args()

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        print("No file uploaded")
        response = jsonify({"message": "No file uploaded"})
        response.status_code = 400
        response.headers.add("Access-Control-Allow-Origin", "*")  # Allow all origins
        return response

    file = request.files['image']
    file_path = f"uploads/{file.filename}"
    file.save(file_path)  # Save the uploaded file

    # in_arg = get_input_args()

    # my_model = tf_keras.models.load_model(my_model, custom_objects={'KerasLayer': hub.KerasLayer})
    # category_path = in_arg.category_names
    # k = 1

    probs, classes = predict(path_image= file_path)
    # probs, classes = predict()
    print(f"Probs: {probs}\nClasses: {classes}")

    print(f"File saved at {file_path}")  # Log file save
    new_line = '\n'
    response = jsonify(f"Species: {classes[0]} - Probability: {(probs[0] * 100):.2f}%")
    response.headers.add("Access-Control-Allow-Origin", "*")  # Allow all origins
    return response

if __name__ == "__main__":
    app.run(debug=True)