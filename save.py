import os, glob
import valohai as vh
import re
import tarfile
import shutil
import zipfile
import time
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# copy the checkpoint files to the new folder
checkpoint_dir = "/home/tensorflow/models/research/new/trained_model"
# os.mkdir("/home/saved_model")
# os.mkdir("/home/saved_model/variables")
os.system(f"mkdir /home/tensorflow/models/research/saved_model")
os.system(f"mkdir /home/tensorflow/models/research/saved_model/variables")
print("***************************")
for path in vh.inputs("saved_model").paths():
    print(path)
    if "saved_model.pb" in path:
        os.system(f"cp {path} /home/tensorflow/models/research/saved_model")
    else:
        os.system(f"cp {path} /home/tensorflow/models/research/saved_model/variables")
print("***************************")
print(os.listdir("/home/tensorflow/models/research/saved_model"))

# load model test

try:
    # Specify the updated path to the SavedModel directory
    saved_model_path = '/home/tensorflow/models/research/saved_model'  # Updated path

    # Load the SavedModel
    loaded_model = tf.saved_model.load(saved_model_path)

    # Select the signature you want to use ('classify' or 'serving_default')
    signature_key = 'classify'  # Replace with 'serving_default' if needed

    # Get the input and output tensor names from the selected signature
    signature = loaded_model.signatures[signature_key]

    # Get the name of the input tensor directly
    input_tensor_name = signature.inputs[0].name
    print("model loaded", input_tensor_name)
except:
    print("failed to load model")