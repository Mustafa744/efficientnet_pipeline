import tensorflow as tf
import numpy as np
import valohai as vh
import os
import json

dataset_info_path = vh.inputs('dataset_info').path()
# read json file
with open(dataset_info_path, 'r') as f:
    dataset_info = json.load(f)
    # print its content
    print("loaded json")
    print(dataset_info)

# save the dataset info to a txt file
with open(vh.outputs().path("dataset_info.txt"), "w") as f:
    f.write(str(dataset_info))
    