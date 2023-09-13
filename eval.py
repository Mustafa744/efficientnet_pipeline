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

import tensorflow as tf
import tensorflow_datasets as tfds

class CustomTFRecordDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release"}

    def _info(self):
        # Define the dataset metadata, including features and labels
        features = {
            'image': tfds.features.Image(shape=(28, 28, 1), dtype=tf.uint8),
            'label': tfds.features.ClassLabel(num_classes=10),
        }
        return tfds.core.DatasetInfo(
            builder=self,
            description="Custom dataset from TFRecords",
            features=tfds.features.FeaturesDict(features),
        )

    def _split_generators(self, dl_manager):
        # Define how to download and extract the dataset
        # You can specify the file paths or URLs for your TFRecord files here

        splits = {
            'train': train_tfrecord_file,
            'validation': validate_tfrecord_file,
            'test': validate_tfrecord_file,
        }

        return {split_name: tfds.core.SplitGenerator(
                name=split_name,
                gen_kwargs={'file_path': file_path},
            ) for split_name, file_path in splits.items()}

    def _generate_examples(self, file_path):
        # Define how to parse the TFRecord files and yield examples
        # Example:
        for record in tf.data.TFRecordDataset(file_path):
            # Parse the record and yield examples
            # Example:
            feature_dict = tf.io.parse_single_example(
                record,
                features={
                    'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.int64),
                }
            )
            image = tf.image.decode_image(feature_dict['image'], channels=1)
            label = feature_dict['label']
            yield {
                'image': image,
                'label': label,
            }

tfds.builder('custom_tfrecord_dataset')
os.system("tfds build custom_tfrecord_dataset")