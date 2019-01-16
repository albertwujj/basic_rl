import tensorflow as tf
from nn_utils.mlp import mlp

def split_net(x, base_sizes, split_sizes, split_count, activation=None, output_activation=None):
    base = mlp(x, base_sizes, activation, activation)
    splits = []
    for _ in range(split_count):
        splits.append(mlp(base, list(split_sizes) + [1], activation, output_activation))
    return tf.concat(splits, -1)