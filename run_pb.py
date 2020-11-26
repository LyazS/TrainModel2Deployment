import tensorflow as tf
import numpy as np

pb_path = "mymodel/detnet.pb"

with tf.gfile.FastGFile(pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    inp, uv, xyz = tf.import_graph_def(graph_def,
                                       return_elements=[
                                           "prior_based_hand/input_0:0",
                                           "output_uv:0",
                                           "output_xyz:0",
                                       ])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result1, result2 = sess.run([uv, xyz],
                                feed_dict={inp: np.random.rand(128, 128, 3)})
    print(result1.shape, result2.shape)
