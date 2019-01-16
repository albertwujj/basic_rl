import tensorflow as tf

def mlp(x, layer_sizes, activation=None, output_activation=None):
    for s in layer_sizes[:-1]:
        x = tf.layers.dense(x, s, activation=activation)
    return tf.layers.dense(x, layer_sizes[-1], activation=output_activation)