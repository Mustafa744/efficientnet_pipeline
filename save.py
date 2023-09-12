# import os, glob
# import valohai as vh
# import re
# import tarfile
# import shutil
# import zipfile
# import time
# import tensorflow as tf

# tf.compat.v1.disable_eager_execution()

# # copy the checkpoint files to the new folder
# checkpoint_dir = "/home/tensorflow/models/research/new/trained_model"
# # os.mkdir("/home/saved_model")
# # os.mkdir("/home/saved_model/variables")
# os.system(f"mkdir /home/tensorflow/models/research/saved_model")
# os.system(f"mkdir /home/tensorflow/models/research/saved_model/variables")
# print("***************************")
# for path in vh.inputs("saved_model").paths():
#     print(path)
#     if "saved_model.pb" in path:
#         os.system(f"cp {path} /home/tensorflow/models/research/saved_model")
#     else:
#         os.system(f"cp {path} /home/tensorflow/models/research/saved_model/variables")
# print("***************************")
# print(os.listdir("/home/tensorflow/models/research/saved_model"))
# print("***************************")
# print(os.listdir("/home/tensorflow/models/research/saved_model/variables"))

# # load model test

# try:
#     # Specify the updated path to the SavedModel directory
#     saved_model_path = '/home/tensorflow/models/research/saved_model'  # Updated path

#     # Load the SavedModel
#     loaded_model = tf.saved_model.load(saved_model_path)

#     # Select the signature you want to use ('classify' or 'serving_default')
#     signature_key = 'classify'  # Replace with 'serving_default' if needed

#     # Get the input and output tensor names from the selected signature
#     signature = loaded_model.signatures[signature_key]

#     # Get the name of the input tensor directly
#     input_tensor_name = signature.inputs[0].name
#     print("model loaded", input_tensor_name)
# except Exception as e:
#     print("failed to load model",e)
# ###########################################

import tensorflow as tf
import numpy as np
import valohai as vh
import os
from PIL import Image
import io


# tf.compat.v1.disable_eager_execution()
saved_model_path = '/home/tensorflow/models/research/saved_model'  # Updated path
# load the tf model from the previous step
# prepare the saved model folder
os.system(f"mkdir /home/tensorflow/models/research/saved_model")
os.system(f"mkdir /home/tensorflow/models/research/saved_model/variables")
for path in vh.inputs("saved_model").paths():
    print(path)
    if "saved_model.pb" in path:
        os.system(f"cp {path} /home/tensorflow/models/research/saved_model")
    else:
        os.system(f"cp {path} /home/tensorflow/models/research/saved_model/variables")

# Example usage:
label_map_path = vh.inputs("labels_map").path()  # Replace with your file path
# label_map = load_label_map(label_map_path)
saved_model_path = '/home/tensorflow/models/research/saved_model'  # Updated path
# tfrecord_path = "gs://valohai_object_detection/data/01FZC/01FZCRSPP49MRZ3XTZE8Q8BN3V/output-315/trained/efficientnet/validation-0.tfrecord-00000-of-00001"
tfrecord_path = vh.inputs("tf_record").path()
def load_model(saved_model_path, signature_key='classify'):
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
    return loaded_model, signature

def predict_with_model(model, signature, image_bytes):
    """Predict the class of an image using a loaded TensorFlow model.

    Args:
        model (tf.saved_model.SavedModel): The loaded TensorFlow model.
        signature (tf.function): The selected signature function.
        image_bytes (bytes): Serialized image data.

    Returns:
        dict: Predicted classes and probabilities.
    """
    # Load and preprocess the image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    input_data_bytes = image_bytes.getvalue()

    # Prepare input tensor as a string tensor
    input_tensor = tf.constant([input_data_bytes], dtype=tf.string)

    # Make a prediction using the signature
    predictions = signature(input_tensor)

    # Extract the output tensors from the predictions
    output_classes = predictions['classes']
    output_probabilities = predictions['probabilities']

    # Return the predicted classes and probabilities
    return {"Predicted Classes": output_classes, "Predicted Probabilities": output_probabilities}

def parse_tfrecord(tfrecord_path, saved_model_path, signature_key='classify'):
    """Parse a TFRecord file and make predictions on its contents.

    Args:
        tfrecord_path (str): Path to the TFRecord file.
        saved_model_path (str): Path to the SavedModel directory.
        signature_key (str): The signature key to use ('classify' or 'serving_default').

    Returns:
        list: List of dictionaries with predicted classes and probabilities for each record.
    """
    # Load the model and signature
    model, signature = load_model(saved_model_path, signature_key)

    # Create a TFRecord dataset from the file
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    predictions_list = []

    # Iterate through the dataset and make predictions
    for record in dataset:
        try:
            # Parse the TFRecord example
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            # Print the TFRecord content for debugging
            # print("TFRecord Content:")
            # print(example)
            # write to txt file
            with open(vh.outputs().path("tf.txt"), "w") as f:
                f.write(str(example))

            # Check if 'image' feature exists in the example
            if 'image' not in example.features.feature:
                continue  # Skip this record and move to the next one

            # Extract image data
            image_bytes = example.features.feature['image'].bytes_list.value[0]

            # Make predictions on the image
            predictions = predict_with_model(model, signature, image_bytes)

            # Append the predictions to the list
            predictions_list.append(predictions)

        except IndexError:
            print("Error: 'image' feature not found or empty in the TFRecord. Skipping this record.")
            continue

    return predictions_list

predictions = parse_tfrecord(tfrecord_path, saved_model_path)

# Print the predictions (you can process them further as needed)
for idx, prediction in enumerate(predictions, start=1):
    print(f"Prediction for Example {idx}: {prediction}")



