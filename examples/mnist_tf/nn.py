import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope, arg_scope


'''
basic building blocks
'''

def make_model(name, template, **kwargs):
    """Create model with fixed kwargs."""
    run = lambda *args, **kw: template(*args, **dict((k, v) for kws in (kw, kwargs) for k, v in kws.items()))
    return tf.make_template(name, run, unique_name_ = name)


@add_arg_scope
def conv2D(x, f, k, s=1, padding="same", activation=None):
    return tf.layers.Conv2D(f, k, strides=s, padding=padding, activation=activation)(x)


@add_arg_scope
def dense(x, f, activation=None):
    return tf.layers.dense(x, f, activation=activation)