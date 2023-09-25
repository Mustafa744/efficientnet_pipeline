import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Specify the updated path to the SavedModel directory
saved_model_path = '/home/saved'  # Updated path

# Load the SavedModel
loaded_model = tf.saved_model.load(saved_model_path)

# Select the signature you want to use ('classify' or 'serving_default')
signature_key = 'classify'  # Replace with 'serving_default' if needed

# Get the input and output tensor names from the selected signature
signature = loaded_model.signatures[signature_key]

# Get the name of the input tensor directly
input_tensor_name = signature.inputs[0].name

# Generate a random JPEG image as input data
random_image = Image.new('RGB', (224, 224))
random_image_bytes = io.BytesIO()
random_image.save(random_image_bytes, format='JPEG')
input_data_bytes = random_image_bytes.getvalue()

# Prepare input tensor as a string tensor
input_tensor = tf.constant([input_data_bytes], dtype=tf.string)

# Make a prediction using the 'predict' method from the signature
predictions = signature(input_tensor)

# Extract the output tensors from the predictions
output_classes = predictions['classes']
output_probabilities = predictions['probabilities']

# Use the predicted classes and probabilities for further processing
print("Predicted Classes:", output_classes)
print("Predicted Probabilities:", output_probabilities)
