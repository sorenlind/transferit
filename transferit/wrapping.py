"""
Functions for wrapping up a model for serving with TF Serving.

This code was heavily inspired by this blog post:
https://medium.com/analytics-vidhya/71d58316570c
"""

from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import get_session  # type: ignore
from tensorflow.compat.v1.saved_model import signature_constants as sig_consts_v1  # type: ignore # noqa
from tensorflow.compat.v1.saved_model.utils import build_tensor_info  # type: ignore
from tensorflow.keras.models import load_model
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops, lookup_ops
from tensorflow.python.saved_model import builder as saved_builder
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import (
    build_signature_def,
    is_valid_signature,
)

IMAGE_SIZE = (256, 256)


def wrap_model(
    model_file: Path,
    output_folder: Path,
    class_names: List[str],
    top_k: Optional[int],
):
    tf.compat.v1.disable_eager_execution()

    if model_file.suffix != ".hdf5":
        raise ValueError("Model file must be saved as hdf5.")

    model = load_model(model_file, compile=False)
    n_classes = int(model.output.shape[-1])  # type: ignore
    print(f" ◦  {n_classes} classes")
    if n_classes != len(class_names):
        raise ValueError("Number of specified classes does not match model shape")

    builder = saved_builder.SavedModelBuilder(str(output_folder))

    classification_signature = _create_signature(model, class_names, top_k)
    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature
    }
    session = get_session()
    builder.add_meta_graph_and_variables(
        session,
        [tag_constants.SERVING],
        signature_def_map=signature_def_map,
        main_op=tf.compat.v1.tables_initializer(),
        strip_default_attrs=True,
    )
    builder.save()


def _preprocess_image(image_buffer):
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Add bias as per ResNet 50 preprocessing function
    red_bias = -123.68
    green_bias = -116.779
    blue_bias = -103.939
    image = tf.math.add(image, [red_bias, green_bias, blue_bias])

    # Make BGR
    image = tf.reverse(image, axis=[-1])
    return image


def _create_signature(model, class_names: List[str], top_k: Optional[int]):
    serialized_tf_example = array_ops.placeholder(tf.string, name="tf_example")
    feature_configs = {"x": tf.io.FixedLenFeature([], tf.string)}
    tf_example = tf.io.parse_example(serialized_tf_example, feature_configs)
    jpegs = tf_example["x"]
    x = tf.map_fn(_preprocess_image, jpegs, dtype=tf.float32)
    y = model(x)

    top_k = min(top_k or len(class_names), len(class_names))
    values, indices = tf.nn.top_k(y, top_k)
    table_class_names = lookup_ops.index_to_string_table_from_tensor(
        vocabulary_list=tf.constant(class_names), default_value="UNK", name=None
    )
    classification_inputs = build_tensor_info(serialized_tf_example)
    prediction_class_names = table_class_names.lookup(
        tf.cast(indices, dtype=dtypes.int64)
    )
    classification_outputs_class_names = build_tensor_info(prediction_class_names)
    classification_outputs_scores = build_tensor_info(values)
    classification_signature = build_signature_def(
        inputs={sig_consts_v1.CLASSIFY_INPUTS: classification_inputs},
        outputs={
            sig_consts_v1.CLASSIFY_OUTPUT_CLASSES: classification_outputs_class_names,
            sig_consts_v1.CLASSIFY_OUTPUT_SCORES: classification_outputs_scores,
        },
        method_name=sig_consts_v1.CLASSIFY_METHOD_NAME,
    )

    # Ensure valid signature
    if not is_valid_signature(classification_signature):
        raise ValueError("Invalid classification signature!")

    return classification_signature
