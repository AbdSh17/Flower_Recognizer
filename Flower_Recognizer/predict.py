import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import tf_keras
import argparse
import tensorflow_hub as hub
from PIL import Image
import json

# Default paths
my_image_path = './test_images/hard-leaved_pocket_orchid.jpg'
my_model = tf_keras.models.load_model("my_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})


def process_image(image):
    pre_image_size = 224

    image = tf.convert_to_tensor(image, dtype=tf.float32) # Convert the datatype to float
    image = tf.image.resize(image, (pre_image_size, pre_image_size)) # Convert the image to (224, 224, 3)
    image /= 255 # rescale the image from scope 0-255 -> 0-1
    image = image.numpy() # re-convert it to numpy array
    return image

def predict(path_image= my_image_path, model=my_model, names_path= "label_map.json", k=5):
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

    parser.add_argument('image_path', type=str, default='./test_images/hard-leaved_pocket_orchid.jpg', help='path to the images')
    parser.add_argument('model_path', type=str, default='my_model.h5',help='path to the model')
    parser.add_argument('--top_k', type=int, default=5, help='number of returned classes along side with their probs')
    parser.add_argument('--category_names', type=str, default='label_map.json', help='category_names')

    return parser.parse_args()

if __name__ == "__main__":

    in_arg = get_input_args()

    image_path = in_arg.image_path
    my_model = tf_keras.models.load_model(in_arg.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    category_path = in_arg.category_names
    k = in_arg.top_k

    probs, classes = predict(image_path, my_model, category_path, k)
    print(f"Probs: {probs}\nClasses: {classes}")