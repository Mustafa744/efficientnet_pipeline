import json
import tensorflow as tf
import tensorflow_datasets as tfds
import valohai as vh

# Define the paths to your TFRecord files using Valohai inputs
train_tfrecord_file = vh.inputs("train").path()
validate_tfrecord_file = vh.inputs("validate").path()
test_tfrecord_file = vh.inputs("test").path()


def calculate_metadata(tfrecord_path):
    # Create a TFRecord dataset from the file
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Initialize an empty dictionary to store feature information
    features = {}

    # Iterate through the dataset to extract feature information
    for record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        for feature_name, feature in example.features.feature.items():
            dtype = tf.dtypes.as_dtype(feature.WhichOneof('kind')).name
            shape = feature.ListFields()[0][1].shape

            features[feature_name] = {
                'pythonClassName': 'tensorflow_datasets.core.features.feature.Feature',
                feature_name: {
                    'dtype': dtype,
                    'shape': {
                        'dimensions': [str(dim) for dim in shape],
                    },
                },
            }

    return features

def generate_metadata(tfrecord_path, output_path):
    # Calculate metadata for the TFRecord file
    metadata = {
        'pythonClassName': 'tensorflow_datasets.core.features.features_dict.FeaturesDict',
        'featuresDict': {
            'features': calculate_metadata(tfrecord_path),
        },
    }
    with open(output_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)



# Example usage:
tfrecord_path = train_tfrecord_file
output_path = vh.outputs().path('metadata.json')
generate_metadata(tfrecord_path, output_path)

