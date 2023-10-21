import tensorflow as tf
import valohai as vh
import os
import csv
import cv2
import numpy as np
import re


# -------------------- Write/Read TF record logic --------------------
class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg = tf.io.decode_image

    def decode_jpeg(self, image_data):
        image = self._decode_jpeg(image_data, channels=3)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def read_label_map(label_map_path):
    with open(label_map_path, "r") as f:
        label_map_string = f.read()
    # regex pattern to extract id and name values
    pattern = r"id:\s*(\d+)\s*\n\s*name:\s*'([^']*)'"

    # extract id and name values using regex
    result = dict(re.findall(pattern, label_map_string))

    # print the resulting dictionary
    return result


coder = ImageCoder()
csv_path = vh.inputs("csv_path").path()
tfrecords_name = "dataset.tfrecord"
label_map_path = vh.inputs("label_map_path").path()
label_map = read_label_map(label_map_path)


def parse_single_example(serialized_example):
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/colorspace": tf.io.FixedLenFeature([], tf.string),
            "image/channels": tf.io.FixedLenFeature([], tf.int64),
            "image/format": tf.io.FixedLenFeature([], tf.string),
            "image/filename": tf.io.FixedLenFeature([], tf.string),
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/class/label": tf.io.FixedLenFeature([], tf.int64),
            "image/class/text": tf.io.FixedLenFeature([], tf.string),
        },
    )
    image = tf.image.decode_image(features["image_raw"], channels=3)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.cast(features["label"], tf.int32)
    return image, label


def create_tfrecord(csv_path, tfrecords_name):
    # Open the CSV file
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)

        # Skip the header row
        next(reader)

        # Create a TFRecord writer
        writer = tf.io.TFRecordWriter(tfrecords_name)

        # Loop through each row in the CSV file
        for row in reader:
            # print(row)
            # Extract the image path and label from the row
            image_path = row[1]
            label = int(row[2])
            label_text = label_map[str(label)]

            # Read the image data
            with tf.io.gfile.GFile(image_path, "rb") as f:
                image_data = f.read()
            # Decode the JPEG image data
            image = coder.decode_jpeg(image_data)
            # Create a TFRecord example
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image/encoded": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image_data])
                        ),
                        "image/class/label": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label])
                        ),
                        "image/class/text": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[label_text.encode("utf-8")]
                            )
                        ),
                        "image/filename": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[image_path.encode("utf-8")]
                            )
                        ),
                        "image/format": tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[image_path.split(".")[-1].encode("utf-8")]
                            )
                        ),
                        "image/colorspace": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[b"rgb"])
                        ),
                        "image/channels": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[3])
                        ),
                        "image/height": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[224])
                        ),
                        "image/width": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[224])
                        ),
                    }
                )
            )

            # Write the example to the TFRecord file
            writer.write(example.SerializeToString())

        # Close the TFRecord writer
        writer.close()


def read_tfrecord(tfrecord_path):
    # Define the feature description
    feature_description = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/colorspace": tf.io.FixedLenFeature([], tf.string),
        "image/channels": tf.io.FixedLenFeature([], tf.int64),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/class/label": tf.io.FixedLenFeature([], tf.int64),
        "image/class/text": tf.io.FixedLenFeature([], tf.string),
    }

    # Create a dataset from the TFRecord file
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Parse the serialized data in the TFRecord file
    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(_parse_function)

    # Print the features of each example
    for example in dataset:
        label = example["image/class/label"]
        img_array = coder.decode_jpeg(example["image/encoded"])
        print("label ", label, "image: ", img_array.shape)
        cv2.imwrite(f"label_{label}.png", np.array(img_array))


# create tfrecord
create_tfrecord(csv_path, tfrecords_name)
# Read and iterate over the TFRecord file
read_tfrecord(vh.outputs().path("dataset.tfrecord"))
