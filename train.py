import os, glob
import valohai as vh
import tarfile
import shutil
import zipfile
import time
from threading import Thread

data_dir = vh.inputs("tfrecords").path()
data_dir = data_dir.split("/")
data_dir = data_dir[:-1]
data_dir = "/".join(data_dir)
data_dir = f"{data_dir}/"

use_tpu = vh.parameters("use_tpu").value
train_steps = vh.parameters("train_steps").value
# model_dir = vh.parameters('model_dir').value
num_label_classes = vh.parameters("num_label_classes").value
train_batch_size = vh.parameters("train_batch_size").value

print(data_dir)
os.system(
    f"""python /home/tensorflow/models/research/models/official/efficientnet/main.py --use_tpu=False --data_dir={data_dir} --model_dir=/home/tensorflow/models/research/new/ --train_steps=10 --skip_host_call=true --num_label_classes=5 --train_batch_size=8 """
)


path = vh.outputs().path("trained")
shutil.make_archive(path, "zip", "/home/tensorflow/models/research/new/")
with vh.metadata.logger() as logger:
    logger.log("model", path)
    logger.log("listing temp directory", os.listdir("/tmp"))
    if "labels_map.pbtxt" in os.listdir("/tmp"):
        logger.log("found labels_map.pbtxt")
        os.mv("/tmp/labels_map.pbtxt", vh.outputs().path("labels_map.pbtxt"))
    
    elif "labels_map.txt" in os.listdir("/tmp"):
        os.mv("/tmp/labels_map.txt", vh.outputs().path("labels_map.txt"))
        

os.system(
    f"cp /home/tensorflow/models/research/new/ {vh.outputs('tfrecord').path('efficientnet')}"
)
