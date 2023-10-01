import tensorflow as tf
import valohai as vh

# use tf eager execution
tf.compat.v1.enable_eager_execution()

class HandleTFRecord:
    image_id = 0
    def __init__(self, tfrecord_path):
        self.tfrecord_path = tfrecord_path

    
    def get_tfrecord_images(self,tfrecord_path):
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        for i, record in enumerate(dataset):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
        for feature_name, feature in example.features.feature.items():
                if feature.HasField('bytes_list'):
                    # Assume the bytes_list contains image data in raw format
                    image_bytes = feature.bytes_list.value[0]
                    
                    # Decode the raw bytes to an image tensor (Assuming RGB images)
                    image = tf.io.decode_image(image_bytes, channels=3)
                    
                    # You can do further processing here if needed, like resizing or normalization
                    
                    # Save the image to disk
                    image_filename = f'image_{i}.png'  # You can change the filename as per your requirement
                    image_path = vh.outputs().path(image_filename)
                    tf.keras.preprocessing.image.save_img(image_path, image)
                    print(f'Saved image {i} to {image_path}')
        # for record in enumerate(dataset):
        #     example = tf.train.Example()
        #     example.ParseFromString(record.numpy())

        #     for feature_name, feature in example.features.feature.items():
        #         if feature.HasField('int64_list'):
        #             dtype = 'int64'
        #             shape = []
        #         elif feature.HasField('float_list'):
        #             dtype = 'float32'
        #             shape = []
        #         elif feature.HasField('bytes_list'):
        #             dtype = 'uint8'
        #             # Decode the bytes_list to determine the actual shape
        #             value = tf.io.decode_raw(feature.bytes_list.value[0], out_type=tf.uint8)
        #             shape = value.shape.as_list()
        #         else:
        #             dtype = 'unknown'
        #             shape = []
        

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