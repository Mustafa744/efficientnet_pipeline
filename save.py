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
os.mkdir("/home/tensorflow/models/research/new/trained_model")
for path in vh.inputs("trained_model").paths():
    os.system(f"cp {path} /home/tensorflow/models/research/new/trained_model")

# test if files were copied
with vh.metadata.logger() as logger:
    logger.log("checkpoint path", checkpoint_dir)
    logger.log("checkpoint directory content: ", os.listdir(checkpoint_dir))


meta_file = checkpoint_dir + "/model.ckpt-0.meta"

saver = tf.compat.v1.train.import_meta_graph(meta_file)
checkpoint = tf.compat.v1.train.latest_checkpoint(checkpoint_dir)

# Create a TensorFlow 1 session
with tf.compat.v1.Session() as sess:
    # Restore the checkpoint variables
    saver.restore(sess, checkpoint)

    # Get the names of all the nodes in the graph
    all_node_names = [
        n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node
    ]

    # Freeze the entire TensorFlow 1 graph
    frozen_graph_def = tf.compat.v1.graph_util.extract_sub_graph(
        sess.graph_def, all_node_names
    )

    # Replace 'frozen_graph.pb' with the desired output pb file path
    pb_file_path = f"""{vh.outputs().path("frozen_graph.pb")}"""
    with tf.io.gfile.GFile(pb_file_path, "wb") as f:
        f.write(frozen_graph_def.SerializeToString())
