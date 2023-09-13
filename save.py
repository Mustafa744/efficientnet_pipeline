import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import datetime
import valohai as vh
import os
import ast


# tf.compat.v1.disable_eager_execution()
saved_model_path = '/home/tensorflow/models/research/saved_model'  # Updated path
# load the tf model from the previous step
# prepare the saved model folder
os.system(f"mkdir /home/tensorflow/models/research/saved_model")
os.system(f"mkdir /home/tensorflow/models/research/saved_model/variables")
for path in vh.inputs("saved_model").paths():
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
    
def load_label_map(label_map_path):
    """Load a label map from a labels_map.pbtxt file.

    Args:
        label_map_path (str): Path to the labels_map.pbtxt file.

    Returns:
        dict: A dictionary mapping class IDs to class names.
    """
    label_map_dict = {}
    
    with tf.io.gfile.GFile(label_map_path, 'r') as f:
        label_map_data = f.read()
        label_map_list = label_map_data.split('item {')[1:]  # Split into individual class definitions
        
        for item in label_map_list:
            item_id = int(item.split('id: ')[1].split('\n')[0])
            item_name = item.split('name: ')[1].split('\n')[0].strip('"')
            label_map_dict[item_id] = item_name

    return label_map_dict

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
    # with vh.metadata.logger() as logger:
    #     logger.log("printing prediction...", predictions)

    # Return the predicted classes and probabilities
    return {"Predicted Classes": output_classes, "Predicted Probabilities": output_probabilities}

def process_prediction(predictions, label_map_dict):
    """Process model predictions and return class label, class name, and probability.

    Args:
        predictions (dict): Model predictions with 'printing prediction...' key containing 'classes' and 'probabilities'.
        label_map_dict (dict): A dictionary mapping class labels to class names.

    Returns:
        dict: Class label, class name, and probability.
    """
    prediction_data = predictions['printing prediction...']
    predicted_classes = prediction_data["classes"].numpy()
    predicted_probabilities = prediction_data["probabilities"].numpy()

    # Assuming 'classes' contains a single class label (tf.Tensor), extract it
    class_label = int(predicted_classes[0])

    # Find the class name using the label map
    class_name = label_map_dict.get(class_label, 'Unknown Class')

    # Assuming 'probabilities' contains a single set of probabilities, extract them
    probabilities = predicted_probabilities[0]

    return {
        "Class Label": class_label,
        "Class Name": class_name,
        "Probability": probabilities[class_label]
    }

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
    for idx, record in enumerate(dataset, start=1):
        try:
            # Parse the TFRecord example
            example = tf.train.Example()
            example.ParseFromString(record.numpy())

            # Extract image data (update feature keys as needed)
            image_encoded = example.features.feature['image/encoded'].bytes_list.value[0]

            # Make predictions on the image
            predictions = predict_with_model(model, signature, image_encoded)
            
            # Process the prediction using the process_prediction function
            processed_prediction = process_prediction(predictions, label_map_dict)

            # Append the processed prediction to the list
            predictions_list.append(processed_prediction)

            # Append the predictions to the list
            predictions_list.append(predictions)

        except Exception as e:
            print(f"Error processing TFRecord {idx}: {e}")
            
    # with vh.metadata.logger() as logger:
    #     logger.log("predictions", predictions_list)
    return predictions_list

def calculate_class_metrics(predictions_list, label_map_dict, confidence_threshold=0.05):
    """Calculate class-wise evaluation metrics for each class based on predictions.

    Args:
        predictions_list (list): List of dictionaries containing predictions for each record.
        label_map_dict (dict): A dictionary mapping class names to class labels.
        confidence_threshold (float): Threshold for considering a prediction as positive.

    Returns:
        dict: Evaluation metrics for each class in the specified format.
    """
    evaluation_time = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    execution_time = 263431  # Replace with the actual execution time
    model_id = "classification-model-1-1684231321135"  # Replace with the actual model ID
    
    evaluation_json = {
        "createTime": evaluation_time,
        "execution_time": execution_time,
        "id": model_id,
        "evaluatedPerClass": {}
    }

    for class_name, class_label in label_map_dict.items():
        class_metrics = {
            "confidenceMetricsEntry": []
        }

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for prediction in predictions_list:
            true_class_label = prediction["Class Label"]
            probability = prediction["Probability"]

            is_positive = probability >= confidence_threshold

            if true_class_label == class_label and is_positive:
                true_positives += 1
            elif true_class_label != class_label and is_positive:
                false_positives += 1
            elif true_class_label == class_label and not is_positive:
                false_negatives += 1

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        class_metrics["confidenceMetricsEntry"].append({
            "confidenceThreshold": confidence_threshold,
            "f1score": f1_score,
            "precision": precision,
            "recall": recall
        })

        evaluation_json["evaluatedPerClass"][class_name] = class_metrics

    return evaluation_json

predictions = parse_tfrecord(tfrecord_path, saved_model_path)

# Load the label map from the provided labels_map.pbtxt file
label_map_path = vh.inputs("labels_map").path()  # Replace with your file path
label_map_dict = load_label_map(label_map_path)

# Specify the path to the SavedModel
saved_model_path = '/home/tensorflow/models/research/saved_model'  # Updated path

# Specify the path to the TFRecord file
tfrecord_path = vh.inputs("tf_record").path()  # Replace with your TFRecord file path

# Parse the TFRecord and make predictions
predictions = parse_tfrecord(tfrecord_path, saved_model_path)

# Calculate class-wise metrics
confidence_threshold = 0.05  # Replace with your desired confidence threshold
evaluation_json = calculate_class_metrics(predictions, label_map_dict, confidence_threshold)

# save the evaluation JSON as json file 
with open(vh.outputs().path("evaluation.json"), 'w') as f:
    json.dump(evaluation_json, f, indent=2)

