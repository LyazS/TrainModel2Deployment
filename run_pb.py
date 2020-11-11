import tensorflow as tf
import numpy as np

pb_path = "mymodel/p2.pb"

with tf.gfile.FastGFile(pb_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    inp, out1, out2, out3 = tf.import_graph_def(graph_def,
                                                return_elements=[
                                                    "input_1:0", "output_1:0",
                                                    "output_2:0", "output_3:0"
                                                ])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result1, result2, result3 = sess.run(
        [out1, out2, out3], feed_dict={inp: np.random.rand(1, 368, 370, 3)})
    print(result1.shape, result2.shape, result3.shape)
