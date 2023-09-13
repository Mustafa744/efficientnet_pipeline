import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import datetime
import valohai as vh
import os
import ast

# Define the path to the saved model
saved_model_path = '/home/tensorflow/models/research/saved_model'

# Create the saved_model folder and copy the model files
os.makedirs(saved_model_path, exist_ok=True)

# Copy saved_model.pb and variables folder from Valohai inputs to the saved_model path
for path in vh.inputs("saved_model").paths():
    if "saved_model.pb" in path:
        os.system(f"cp {path} {saved_model_path}")
    else:
        os.system(f"cp -r {path} {saved_model_path}")

# Define the label_map_path, tfrecord_path, and confidence_threshold
label_map_path = vh.inputs("labels_map").path()
tfrecord_path = vh.inputs("tf_record").path()
confidence_threshold = 0.05

# Load the label map
def load_label_map(label_map_path):
    label_map_dict = {}
    with tf.io.gfile.GFile(label_map_path, 'r') as f:
        label_map_data = f.read()
        label_map_list = label_map_data.split('item {')[1:]
        for item in label_map_list:
            item_id = int(item.split('id: ')[1].split('\n')[0])
            item_name = item.split('name: ')[1].split('\n')[0].strip('"')
            label_map_dict[item_id] = item_name
    return label_map_dict

label_map_dict = load_label_map(label_map_path)

# Load the model
def load_model(saved_model_path, signature_key='classify'):
    loaded_model = tf.saved_model.load(saved_model_path)
    signature = loaded_model.signatures[signature_key]
    return loaded_model, signature

model, signature = load_model(saved_model_path)

# Process a single image
def process_image(image_bytes, signature):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    input_data_bytes = image_bytes.getvalue()
    input_tensor = tf.constant([input_data_bytes], dtype=tf.string)
    predictions = signature(input_tensor)
    return predictions

# Parse the TFRecord and make predictions
def parse_tfrecord(tfrecord_path, model, signature, label_map_dict, confidence_threshold):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    predictions_list = []
    for idx, record in enumerate(dataset, start=1):
        try:
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            image_encoded = example.features.feature['image/encoded'].bytes_list.value[0]
            predictions = process_image(image_encoded, signature)
            processed_prediction = process_prediction(predictions, label_map_dict, confidence_threshold)
            predictions_list.append(processed_prediction)
        except Exception as e:
            print(f"Error processing TFRecord {idx}: {e}")
    return predictions_list

# Process a prediction
def process_prediction(predictions, label_map_dict, confidence_threshold):
    prediction_data = predictions['serving_default']
    predicted_classes = prediction_data["classes"].numpy()
    predicted_probabilities = prediction_data["probabilities"].numpy()
    class_label = int(predicted_classes[0])
    class_name = label_map_dict.get(class_label, 'Unknown Class')
    probabilities = predicted_probabilities[0]
    return {
        "Class Label": class_label,
        "Class Name": class_name,
        "Probability": probabilities[class_label]
    }

# Parse TFRecord and make predictions
predictions = parse_tfrecord(tfrecord_path, model, signature, label_map_dict, confidence_threshold)

# Calculate class-wise metrics
def calculate_class_metrics(predictions, label_map_dict, confidence_threshold):
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
        for prediction in predictions:
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

# Calculate class-wise metrics and save the evaluation JSON
evaluation_json = calculate_class_metrics(predictions, label_map_dict, confidence_threshold)
with open(vh.outputs().path("evaluation.json"), 'w') as f:
    json.dump(evaluation_json, f, indent=2)
