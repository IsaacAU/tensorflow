import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.python.tools import inspect_checkpoint

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    print("v2: %s" % v2.eval())

    
