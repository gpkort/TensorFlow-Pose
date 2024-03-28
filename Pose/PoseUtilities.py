import tensorflow_hub as hub
import tensorflow as tf
from enum import Enum


class ModelType(Enum):
    LIGHTNING = 0,
    THUNDER = 1


def get_module(model_type: ModelType) -> tuple:
    module = None
    input_size = 0

    if model_type == ModelType.LIGHTNING:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif model_type == ModelType.THUNDER:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256

    return module.signatures['serving_default'], input_size


def get_keypoints(input_image, model):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """

    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores
