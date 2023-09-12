import tensorflow as tf
import numpy as np
import valohai as vh
import os

tf.compat.v1.disable_eager_execution()
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

# load model
try:
    model = tf.saved_model.load(saved_model_path)
    signature_key = 'classify'
    signature = model.signatures[signature_key]
    input_tensor_name = signature.inputs[0].name
    print("model loaded", input_tensor_name)
except Exception as e:
    print("failed to load model",e)

def make_prediction(image):
    #TODO:implement the function
    pass

def proccess_tfrecord(tfrecord_path):
    # get images and labels from tfrecord of the test dataset
    pass
# label_map_util.py

import tensorflow as tf

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

# Example usage:
label_map_path = vh.inputs().path("label_map")  # Replace with your file path
label_map = load_label_map(label_map_path)
print("Class ID to Class Name Mapping:")
for class_id, class_name in label_map.items():
    print(f"Class ID: {class_id}, Class Name: {class_name}")

