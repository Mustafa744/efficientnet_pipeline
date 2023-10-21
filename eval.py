import tensorflow as tf

# use tf eager execution
tf.compat.v1.enable_eager_execution()
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import save_img


class HandleTFRecord:
    image_id = 0
    images = 0

    def __init__(self, tfrecord_path, model_path):
        self.tfrecord_path = tfrecord_path
        self.model_path = model_path

    def get_tfrecord_images(self, tfrecord_path):
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        # Iterate through the dataset to extract feature information
        for record in dataset:
            HandleTFRecord.images += 1
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            for feature_name, feature in example.features.feature.items():
                print(feature_name, feature)
                if feature.HasField("int64_list"):
                    dtype = "int64"
                    shape = []
                elif feature.HasField("float_list"):
                    dtype = "float32"
                    shape = []
                elif feature.HasField("bytes_list"):
                    dtype = "uint8"
                    # Decode the bytes_list to determine the actual shape
                    value = tf.io.decode_raw(
                        feature.bytes_list.value[0], out_type=tf.uint8
                    )
                    shape = value.shape.as_list()
                else:
                    dtype = "unknown"
                    shape = []
        print("images in tfrecord: ", HandleTFRecord.images)

    def get_tfrecord_sample(self, tfrecord_path):
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        # Iterate through the dataset to extract feature information
        for record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            for feature_name, feature in example.features.feature.items():
                if feature_name != "image/encoded":
                    print(feature_name, feature)
                if feature.HasField("int64_list"):
                    dtype = "int64"
                    shape = []
                elif feature.HasField("float_list"):
                    dtype = "float32"
                    shape = []
                elif feature.HasField("bytes_list"):
                    dtype = "uint8"
                    # Decode the bytes_list to determine the actual shape
                    value = tf.io.decode_raw(
                        feature.bytes_list.value[0], out_type=tf.uint8
                    )
                    shape = value.shape.as_list()
                else:
                    dtype = "unknown"
                    shape = []
        print("images in tfrecord: ", HandleTFRecord.images)

    def load_model(self, saved_model_path, signature_key="classify"):
        """Load a TensorFlow SavedModel and return the model and signature.

        Args:
            saved_model_path (str): Path to the SavedModel directory.
            signature_key (str): The signature key to use ('classify' or 'serving_default').

        Returns:
            tf.saved_model.SavedModel: The loaded model.
            tf.function: The selected signature function.
        """
        loaded_model = tf.saved_model.load(saved_model_path)
        signature = loaded_model.signatures[signature_key]
        self.loaded_model = loaded_model
        self.signature = signature

    def predict_with_model(self, image_bytes):
        """Predict the class of an image using a loaded TensorFlow model.

        Args:
            model (tf.saved_model.SavedModel): The loaded TensorFlow model.
            signature (tf.function): The selected signature function.
            image_bytes (bytes): Serialized image data.

        Returns:
            dict: Predicted classes and probabilities.
        """

        # Decode the image bytes to an image tensor
        image = tf.io.decode_raw(image_bytes, channels=3)
        # Resize the image to the expected shape
        image = tf.image.resize(image, [224, 224])

        save_img(f"{HandleTFRecord.images}.png", image.numpy())
        # Add a batch dimension to the image tensor
        image = tf.expand_dims(image, axis=0)

        raw_image_bytes = io.BytesIO(image_bytes)
        input_data_bytes = raw_image_bytes.getvalue()

        # Prepare input tensor as a string tensor
        input_tensor = tf.constant([input_data_bytes], dtype=tf.string)
        # Make a prediction using the signature
        predictions = self.signature(input_tensor)

        # Extract the output tensors from the predictions
        output_classes = predictions["classes"]
        output_probabilities = predictions["probabilities"]

        # Return the predicted classes and probabilities
        return {
            "Predicted Classes": output_classes,
            "Predicted Probabilities": output_probabilities,
        }

    def process_prediction(self, prediction):
        """Process model predictions and return class label, class name, and probability.

        Args:
            predictions (dict): Model predictions with 'printing prediction...' key containing 'classes' and 'probabilities'.
            label_map_dict (dict): A dictionary mapping class labels to class names.

        Returns:
            dict: Class label, class name, and probability.
        """
        predicted_classes = prediction["Predicted Classes"].numpy()
        predicted_probabilities = prediction["Predicted Probabilities"].numpy()
        global label_map_dict
        # Assuming 'classes' contains a single class label (tf.Tensor), extract it
        class_label = int(predicted_classes[0])

        # Find the class name using the label map
        # class_name = label_map_dict.get(class_label, 'Unknown Class')

        # Assuming 'probabilities' contains a single set of probabilities, extract them
        probabilities = predicted_probabilities[0]

        return {
            "Predicted Label": class_label,
            # "Predicted Class Name": class_name,
            "Probability": probabilities[class_label],
        }

    def predict_sample(self, tfrecord_path, samples=10):
        HandleTFRecord.images = 0
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        # Iterate through the dataset to extract feature information
        print("{")
        for record in dataset.take(samples):
            HandleTFRecord.images += 1
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            image_encoded = example.features.feature["image/encoded"].bytes_list.value[
                0
            ]
            label = example.features.feature["image/class/label"].int64_list.value[0]
            prediction = self.predict_with_model(image_encoded)
            # print(f"true label for {example.features.feature['image/filename'].bytes_list.value[0]}: ", label)
            # print("predicted label: ", self.process_prediction(prediction))
            print(
                f'{example.features.feature["image/filename"].bytes_list.value[0]} : ["predicted_label":{self.process_prediction(prediction)},"true_label":{label}]'
            )
        print("}")

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
# test_tfrecord = vh.inputs("test").path()
# output_path = vh.outputs().path("testtt.tfrecord")
test_tfrecord = "./test.tfrecord"
output_path = "./testtt.tfrecord"
model_path = "./saved"

handler = HandleTFRecord(test_tfrecord, model_path)
# handler.copy_tfrecord(test_tfrecord, output_path)
handler.load_model("./saved")
# handler.predict_sample(test_tfrecord)
handler.get_tfrecord_sample(test_tfrecord)
