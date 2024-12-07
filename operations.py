import tensorflow as tf 
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp

saver = tf.train.import_meta_graph('./model/googlenet.ckpt.meta')
with tf.Session() as sess:
    saver.restore(sess, './model/googlenet.ckpt')
    graph = tf.get_default_graph()
    for op in graph.get_operations():
        print("Operation Name: ", op.name)
        print("Operation Type: ", op.type)