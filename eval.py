import tensorflow as tf
import numpy as np
import valohai as vh
import os
import json
from tensorflow.io import FixedLenFeature

# dataset_info_path = vh.inputs('dataset_info').path()
# # read json file
# with open(dataset_info_path, 'r') as f:
#     dataset_info = json.load(f)
#     # print its content
#     print("loaded json")
#     print(dataset_info)

# # save the dataset info to a txt file
# with open(vh.outputs().path("dataset_info.txt"), "w") as f:
#     f.write(str(dataset_info))

# Define the paths to your TFRecord files
import tensorflow as tf
import tensorflow_datasets as tfds
import json

# # Define the paths to your TFRecord files
# train_tfrecord_file = 'gs://valohai_object_detection/data/01FZC/01FZCRSPP49MRZ3XTZE8Q8BN3V/output-315/trained/efficientnet/train-0.tfrecord-00000-of-00001'
# validate_tfrecord_file = 'gs://valohai_object_detection/data/01FZC/01FZCRSPP49MRZ3XTZE8Q8BN3V/output-315/trained/efficientnet/validation-0.tfrecord-00000-of-00001'
# test_tfrecord_file = 'gs://valohai_object_detection/data/01FZC/01FZCRSPP49MRZ3XTZE8Q8BN3V/output-315/trained/efficientnet/test-0.tfrecord-00000-of-00001'

train_tfrecord_file = vh.inputs("train").path()
validate_tfrecord_file = vh.inputs("validate").path()
test_tfrecord_file = vh.inputs("test").path()

# Define the features of your dataset
features = {}
shapes = {}
for key in ['feature1', 'feature2', 'feature3']:
    if key == 'feature1':
        features[key] = tf.io.FixedLenFeature([], dtype=tf.float32)
        shapes[key] = ()
    elif key == 'feature2':
        features[key] = tf.io.FixedLenFeature([], dtype=tf.int64)
        shapes[key] = ()
    elif key == 'feature3':
        features[key] = tf.io.FixedLenFeature([], dtype=tf.string)
        shapes[key] = ()

# Load a single example from each of the three TFRecord files to determine the features
train_dataset = tf.data.TFRecordDataset(train_tfrecord_file)
train_example = next(train_dataset.take(1).as_numpy_iterator())
parsed_train_example = tf.io.parse_single_example(train_example, features)

validate_dataset = tf.data.TFRecordDataset(validate_tfrecord_file)
validate_example = next(validate_dataset.take(1).as_numpy_iterator())
parsed_validate_example = tf.io.parse_single_example(validate_example, features)

test_dataset = tf.data.TFRecordDataset(test_tfrecord_file)
test_example = next(test_dataset.take(1).as_numpy_iterator())
parsed_test_example = tf.io.parse_single_example(test_example, features)

# Define the shapes of your dataset
shapes = {}
for key in features.keys():
    shapes[key] = ()

# Save the features to a JSON file
with open(vh.outputs().path('features.json',), 'w') as f:
    json.dump({'features': features, 'shapes': shapes}, f)