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

# Create a function to calculate metadata from TFRecord files
def calculate_metadata(tfrecord_file):
    features = {}
    shapes = {}
    
    # Read TFRecord files to calculate metadata
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    for record in dataset.take(1):
        example = tf.io.parse_single_example(record, features)
        for feature_name, feature_tensor in example.items():
            features[feature_name] = feature_tensor.dtype.name
            shapes[feature_name] = feature_tensor.shape.as_list()
    
    return features, shapes

# Calculate metadata for train, validate, and test TFRecord files
train_features, train_shapes = calculate_metadata(train_tfrecord_file)
validate_features, validate_shapes = calculate_metadata(validate_tfrecord_file)
test_features, test_shapes = calculate_metadata(test_tfrecord_file)

# Construct the metadata dictionary
metadata = {
    'pythonClassName': 'tensorflow_datasets.core.features.features_dict.FeaturesDict',
    'featuresDict': {
        'features': {
            feature_name: {
                'pythonClassName': 'tensorflow_datasets.core.features.feature.Feature',
                feature_name: {
                    'shape': {
                        'dimensions': [str(dim) for dim in shape]
                    },
                    'dtype': dtype,
                }
            }
            for feature_name, dtype, shape in zip(
                train_features.keys(), train_features.values(), train_shapes.values()
            )
        }
    }
}

# Save the metadata to a JSON file
with open(vh.outputs().path('metadata.json'), 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=4)

# Print the generated metadata
print(json.dumps(metadata, indent=4))
