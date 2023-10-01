import valohai as vh
import tensorflow as tf
import numpy as np
import cv2
import os
# load test tf record from valohai inputs
test_tfrecord = vh.inputs("test").path()


class HandleTFRecord :
    def __init__(self,tfrecord_path:str) -> None:
        self.tfrecord_path = tfrecord_path
        
    

    def get_tfrecord_images(self):
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=self.tfrecord_path)
        print("***************************")
        # Iterate over all records in the TFRecord file
        for i, string_record in enumerate(record_iterator):
            # Parse the next example
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # Get the image data and label from the example
            image_data = example.features.feature['image'].bytes_list.value[0]
            label = example.features.feature['label'].int64_list.value[0]

            # Get the height, width, and channels of the image
            height = example.features.feature['height'].int64_list.value[0]
            width = example.features.feature['width'].int64_list.value[0]
            channels = example.features.feature['channels'].int64_list.value[0]

            # Convert the image data to a NumPy array
            image = np.frombuffer(image_data, dtype=np.uint8)

            # Reshape the image array to its original shape
            image = image.reshape((height, width, channels))

            # Save the image to disk
            filename = f'image_{label}_{i}.jpg'
            output_path = vh.outputs().path(filename)
            cv2.imwrite(output_path, image)

            # Yield the image and label
            yield image, label
        
    def copy_tfrecord(input_path, output_path):
        # Open the input TFRecord file
        record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=input_path)

        # Create a new TFRecord file for the output
        writer = tf.python_io.TFRecordWriter(output_path)

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
                
                

handler = HandleTFRecord(test_tfrecord)
output_path = vh.outputs().path("testtt_tfrecord.tfrecord")
handler.copy_tfrecord(test_tfrecord, output_path)