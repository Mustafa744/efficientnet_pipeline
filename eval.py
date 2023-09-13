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

# Define the paths to your TFRecord files
train_tfrecord_file = 'gs://valohai_object_detection/data/01FZC/01FZCRSPP49MRZ3XTZE8Q8BN3V/output-315/trained/efficientnet/train-0.tfrecord-00000-of-00001'
validate_tfrecord_file = 'gs://valohai_object_detection/data/01FZC/01FZCRSPP49MRZ3XTZE8Q8BN3V/output-315/trained/efficientnet/validation-0.tfrecord-00000-of-00001'
test_tfrecord_file = 'gs://valohai_object_detection/data/01FZC/01FZCRSPP49MRZ3XTZE8Q8BN3V/output-315/trained/efficientnet/test-0.tfrecord-00000-of-00001'

# Load a single example from the training dataset to determine the features
example = next(tf.data.TFRecordDataset(train_tfrecord_file).take(1).as_numpy_iterator())
features = {}
for key in example.keys():
    if example[key].dtype == 'float32':
        features[key] = tf.io.FixedLenFeature([], dtype=tf.float32)
    elif example[key].dtype == 'int64':
        features[key] = tf.io.FixedLenFeature([], dtype=tf.int64)

# Define the shapes of your dataset
shapes = {}
for key in features.keys():
    shapes[key] = ()

# Load the training dataset
train_dataset = tf.data.TFRecordDataset(train_tfrecord_file)
train_dataset = train_dataset.map(lambda x: tf.io.parse_single_example(x, features))

# Load the validation dataset
validate_dataset = tf.data.TFRecordDataset(validate_tfrecord_file)
validate_dataset = validate_dataset.map(lambda x: tf.io.parse_single_example(x, features))

# Load the testing dataset
test_dataset = tf.data.TFRecordDataset(test_tfrecord_file)
test_dataset = test_dataset.map(lambda x: tf.io.parse_single_example(x, features))

# Create a `tf.data.Dataset` object for each split
train_dataset = train_dataset.batch(32)
validate_dataset = validate_dataset.batch(32)
test_dataset = test_dataset.batch(32)

# Create a `tfds.features.FeaturesDict` object for the features
features_dict = tfds.features.FeaturesDict(features)

# Create a `tfds.core.DatasetInfo` object for the dataset
dataset_info = tfds.core.DatasetInfo(
    builder=tfds.core.DatasetBuilder(),
    description='My dataset description',
    features=features_dict,
    homepage='https://mydataset.com',
    citation='My dataset citation',
    metadata=tfds.core.MetadataDict(),
    supervised_keys=('input', 'label'),
    splits=tfds.core.SplitDict({
        'train': tfds.core.SplitInfo(num_examples=len(train_dataset)),
        'validation': tfds.core.SplitInfo(num_examples=len(validate_dataset)),
        'test': tfds.core.SplitInfo(num_examples=len(test_dataset)),
    }),
    download_checksums={},
)

# Save the dataset info to a JSON file
with open(vh.outputs().path('dataset_info.json'), 'w') as f:
    f.write(dataset_info.as_json)

# Save the features to a JSON file
with open(vh.outputs().path('features.json'), 'w') as f:
    f.write(features_dict.as_json)