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

# Load a single example from each of the three TFRecord files to determine the features
train_example = next(tf.data.TFRecordDataset(train_tfrecord_file).take(1).as_numpy_iterator())
validate_example = next(tf.data.TFRecordDataset(validate_tfrecord_file).take(1).as_numpy_iterator())
test_example = next(tf.data.TFRecordDataset(test_tfrecord_file).take(1).as_numpy_iterator())

# Define the features of your dataset
features = {}
for key in train_example.keys():
    if train_example[key].dtype == 'float32':
        features[key] = tf.io.FixedLenFeature([], dtype=tf.float32)
    elif train_example[key].dtype == 'int64':
        features[key] = tf.io.FixedLenFeature([], dtype=tf.int64)

# Define the shapes of your dataset
shapes = {}
for key in features.keys():
    shapes[key] = ()

# Save the features to a JSON file
with open('features.json', 'w') as f:
    json.dump({'features': features, 'shapes': shapes}, f)