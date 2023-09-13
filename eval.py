import json
import tensorflow as tf
import tensorflow_datasets as tfds
import valohai

# Define the paths to your TFRecord files using Valohai inputs
train_tfrecord_file = valohai.inputs("train").path()
validate_tfrecord_file = valohai.inputs("validate").path()
test_tfrecord_file = valohai.inputs("test").path()

# Create a function to calculate metadata from TFRecord files
def calculate_metadata(tfrecord_file):
    features = {}
    shapes = {}
    
    # Read TFRecord files to determine feature names and data types
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

# Save the metadata to a JSON file at the Valohai output path
with open(valohai.outputs().path('metadata.json'), 'w') as metadata_file:
    json.dump(metadata, metadata_file, indent=4)

# Print the generated metadata
print(json.dumps(metadata, indent=4))
