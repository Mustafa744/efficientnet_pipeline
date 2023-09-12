import tensorflow as tf

# Set the path to the directory containing the saved model files
saved_model_dir = "./saved"

# Start a TensorFlow 1.x session
with tf.compat.v1.Session() as sess:
    # Load the saved model
    model = tf.compat.v1.saved_model.load(sess, ["serve"], saved_model_dir)

    # Print the graph operations and their names
    for op in sess.graph.get_operations():
        print(op.name)

# Close the TensorFlow 1.x session
sess.close()
