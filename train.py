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
os.mkdir("/home/tensorflow/models/research/saved")
os.system(
    f"""python /home/tensorflow/models/research/models/official/efficientnet/main.py --use_tpu=False --data_dir={data_dir} --model_dir=/home/tensorflow/models/research/new/ --train_steps=100 --skip_host_call=true --num_label_classes=5 --train_batch_size=8 --export_dir=/home/tensorflow/models/research/saved"""
)


checkpoint_path = vh.outputs().path("trained")
saved_path = vh.outputs().path("saved")

shutil.make_archive(checkpoint_path, "zip", "/home/tensorflow/models/research/new/")
shutil.make_archive(saved_path, "zip", "/home/tensorflow/models/research/saved")

        

os.system(
    f"cp /home/tensorflow/models/research/new/ {vh.outputs('tfrecord').path('efficientnet')}"
)
