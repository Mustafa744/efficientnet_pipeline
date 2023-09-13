import json
import tensorflow as tf
import tensorflow_datasets as tfds
import valohai as vh

# Define the paths to your TFRecord files using Valohai inputs
train_tfrecord_file = vh.inputs("train").path()
validate_tfrecord_file = vh.inputs("validate").path()
test_tfrecord_file = vh.inputs("test").path()


# Define a dictionary to store metadata
metadata = {
    'pythonClassName': 'tensorflow_datasets.core.features.features_dict.FeaturesDict',
    'featuresDict': {
        'features': {},
    },
}

# Define a function to add feature metadata
def add_feature_metadata(feature_name, dtype, shape):
    metadata['featuresDict']['features'][feature_name] = {
        'pythonClassName': 'tensorflow_datasets.core.features.feature.Feature',
        feature_name: {
            'dtype': dtype,
            'shape': {
                'dimensions': [str(dim) for dim in shape],
            },
        },
    }

# Define a function to calculate metadata from a TFRecord file
def calculate_metadata(tfrecord_path):
    # Create a TFRecord dataset from the file
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Iterate through the dataset to extract feature information
    for record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        for feature_name, feature in example.features.feature.items():
            if feature.HasField('int64_list'):
                dtype = 'int64'
                shape = [len(feature.int64_list.value)]
            elif feature.HasField('float_list'):
                dtype = 'float32'  # Modify as needed
                shape = [len(feature.float_list.value)]
            elif feature.HasField('bytes_list'):
                dtype = 'string'  # Modify as needed
                shape = [len(feature.bytes_list.value)]
            else:
                dtype = 'unknown'
                shape = []

            add_feature_metadata(feature_name, dtype, shape)

# Define a function to generate metadata
def generate_metadata(tfrecord_path, output_path):
    # Calculate metadata for the TFRecord file
    calculate_metadata(tfrecord_path)

    # Save the metadata to a JSON file
    with open(output_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)



# Example usage:
tfrecord_path = train_tfrecord_file
output_path = vh.outputs().path('metadata.json')
generate_metadata(tfrecord_path, output_path)

