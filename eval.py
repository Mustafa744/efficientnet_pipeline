import valohai as vh
import tensorflow as tf

with vh.metadata.logger() as logger:
    logger.log("attempting to load model", vh.inputs("frozen_model").path())
    try:
        # Load the frozen graph
        with tf.io.gfile.GFile("./frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        logger.log("model loaded")
    except Exception as e:
        logger.log("failed to load model", str(e))
