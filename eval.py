import json
import tensorflow as tf
import valohai as vh

# Define the paths to your TFRecord files using Valohai inputs
train_tfrecord_file = vh.inputs("train").path()
validate_tfrecord_file = vh.inputs("validate").path()
test_tfrecord_file = vh.inputs("test").path()

# Define the dataset information dictionary
dataset_info = {
    "description": "Your dataset description here.",
    "citation": "Citation for your dataset, if applicable.",
    "downloadSize": "Unknown",  # Update with the size of your dataset
    "fileFormat": "array_record",
    "location": {
        "urls": ["URL to your dataset homepage, if available"]
    },
    "moduleName": "tensorflow_datasets.image_classification.your_dataset_module",
    "name": "your_dataset_name",
    "splits": [],
    "supervisedKeys": {
        "tuple": {
            "items": [
                {
                    "featureKey": "image"
                },
                {
                    "featureKey": "label"
                }
            ]
        }
    },
    "version": "1.0.0"  # Update with your dataset version
}

# Define a dictionary to store metadata
features_info = {
    'pythonClassName': 'tensorflow_datasets.core.features.features_dict.FeaturesDict',
    'featuresDict': {
        'features': {},
    },
}

# Define a function to add feature metadata
def add_feature_metadata(feature_name, dtype, shape):
    features_info['featuresDict']['features'][feature_name] = {
        'pythonClassName': 'tensorflow_datasets.core.features.feature.Feature',
        feature_name: {
            'dtype': dtype,
            'shape': {
                'dimensions': [str(dim) for dim in shape],
            },
        },
    }

# Define a function to calculate feature metadata from a TFRecord file
def calculate_feature_metadata(tfrecord_path):
    # Create a TFRecord dataset from the file
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Iterate through the dataset to extract feature information
    for record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        for feature_name, feature in example.features.feature.items():
            if feature.HasField('int64_list'):
                dtype = 'int64'
                shape = []
            elif feature.HasField('float_list'):
                dtype = 'float32'
                shape = []
            elif feature.HasField('bytes_list'):
                dtype = 'uint8'
                # Decode the bytes_list to determine the actual shape
                value = tf.io.decode_raw(feature.bytes_list.value[0], out_type=tf.uint8)
                shape = value.shape.as_list()
            else:
                dtype = 'unknown'
                shape = []

            add_feature_metadata(feature_name, dtype, shape)

# Define a function to generate feature metadata JSON
def generate_features_metadata(tfrecord_path, output_path):
    # Calculate metadata for the TFRecord file
    calculate_feature_metadata(tfrecord_path)

    # Save the metadata to a JSON file
    with open(output_path, 'w') as metadata_file:
        json.dump(features_info, metadata_file, indent=4)

# Define a function to calculate split information
def calculate_split_info(split_name, tfrecord_path):
    # Create a TFRecord dataset from the file
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Initialize split info
    split_info = {
        "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
        "name": split_name,
        "numBytes": "Unknown",
        "shardLengths": ["Unknown"]
    }

    # Calculate split size and number of shards
    num_bytes = 0
    num_shards = 0
    for record in dataset:
        num_bytes += len(record.numpy())
        num_shards += 1

    split_info["numBytes"] = str(num_bytes)
    split_info["shardLengths"] = [str(num_shards)]

    return split_info

# Specify the paths to your TFRecord files
train_tfrecord_path = train_tfrecord_file
test_tfrecord_path = test_tfrecord_file
validate_tfrecord_path = validate_tfrecord_file

# Calculate split information for each split
dataset_info["splits"].append(calculate_split_info("train", train_tfrecord_path))
dataset_info["splits"].append(calculate_split_info("test", test_tfrecord_path))
dataset_info["splits"].append(calculate_split_info("validate", validate_tfrecord_path))

# Define a function to generate the dataset_info.json file
def generate_dataset_info(output_path):
    # Save the dataset_info dictionary to a JSON file
    with open(output_path, "w") as dataset_info_file:
        json.dump(dataset_info, dataset_info_file, indent=4)

# Specify the output paths for dataset_info.json and features.json
dataset_info_output_path = vh.outputs().path("dataset_info.json")
features_info_output_path = vh.outputs().path("features.json")

# Generate the dataset_info.json file for the train TFRecord
generate_dataset_info(dataset_info_output_path)
# Generate the features.json file for the train TFRecord
generate_features_metadata(train_tfrecord_path, features_info_output_path)
