import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
import tensorflow as tf
from os import path
from Pose.PoseUtilities import get_module, get_keypoints, ModelType

IMAGE_DIR = "images"

if __name__ == '__main__':
    print('Doodles')

    # Load the input image.
    image_path = 'input_image.jpeg'
    image = tf.io.read_file(path.join(IMAGE_DIR, image_path))
    image = tf.image.decode_jpeg(image)

    model, input_size = get_module(ModelType.THUNDER)

    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    output = get_keypoints(input_image, model)

    print(output.shape)

    # import tensorflow as tf
    # import tensorflow_hub as hub
    #
    #
    # def save_module(url, save_path):
    #     module = hub.KerasLayer(url)
    #     model = tf.keras.Sequential(module)
    #     tf.saved_model.save(model, save_path)
    #
    #
    # save_module("https://tfhub.dev/google/universal-sentence-encoder/4", "./saved-module")

    # https: // www.tensorflow.org / hub / tutorials / movenet
