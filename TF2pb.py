#*-coding:utf-8-*

from keras.models import load_model
from keras import backend
import tensorflow as tf
import os



def tf_to_pb(tf_model, output_dir, model_name, out_prefix="output_"):
    """.h5模型文件转换成pb模型文件
    Argument:
        tf_model: 
            已加载的模型
        output_dir: str
            pb模型文件保存路径
        model_name: str
            pb模型文件名称
        out_prefix: str
            根据训练，需要修改
    Return:
        pb模型文件
    """
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(tf_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(tf_model.output[i], out_prefix + str(i + 1))
    sess = backend.get_session()

    from tensorflow.python.framework import graph_util, graph_io
    # 写入pb模型文件
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)


if __name__ == '__main__':
    
    #  加载模型
    from model_simulated_RGB101 import get_testing_model_resnet101
    # from model_simulated_RGB101_cdcl_pascal import get_testing_model_resnet101
    keras_weights_file = './weights/model_simulated_RGB_mgpu_scaling_append.0071.h5'
    model = get_testing_model_resnet101()
    model.load_weights(keras_weights_file)
    tf_to_pb(model, output_dir="mymodel", model_name="p2.pb")
    print('Finished')