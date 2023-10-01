import tensorflow as tf
import valohai as vh

class HandleTFRecord:
    def __init__(self, tfrecord_path):
        self.tfrecord_path = tfrecord_path

    def get_tfrecord_images(self,tfrecord_path):
        # Create a TFRecordDataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)

        # Define the feature description for the TFRecord
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
        }

        # Parse each example in the dataset
        def _parse_example(example_proto):
            example = tf.io.parse_single_example(example_proto, feature_description)

            # Decode the image data
            image = tf.io.decode_image(example['image'], channels=example['channels'])

            # Reshape the image to its original shape
            image = tf.reshape(image, [example['height'], example['width'], example['channels']])

            # Return the image and label
            return image, example['label']

        # Map the parse function over the dataset
        dataset = dataset.map(_parse_example)

        # Yield each image and label
        for image, label in dataset:
            yield image.numpy(), label.numpy()

    def copy_tfrecord(self, input_path, output_path):
        print("attempting to copy tfrecord")
        # Open the input TFRecord file
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=input_path)

        # Create a new TFRecord file for the output
        writer = tf.io.TFRecordWriter(output_path)

        # Iterate over all records in the input TFRecord file
        for string_record in record_iterator:
            # Parse the next example
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # Serialize the example to a string
            serialized = example.SerializeToString()

            # Write the serialized example to the output TFRecord file
            writer.write(serialized)

        # Close the output TFRecord file
        writer.close()
        print("done copying tfrecord")

# Example usage
test_tfrecord = vh.inputs("test").path()
output_path = vh.outputs().path("testtt.tfrecord")
handler = HandleTFRecord(test_tfrecord)
handler.copy_tfrecord(test_tfrecord, output_path)

handler.get_tfrecord_images(test_tfrecord )