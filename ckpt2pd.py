import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
graph = tf.get_default_graph()

sess = tf.Session()

saver = tf.train.import_meta_graph('./trained_models/tshirt/garment_fit/model.meta')
saver.restore(sess, './trained_models/tshirt/garment_fit/checkpoint')

tf.train.write_graph(sess.graph_def, '.', 'graph.pb', as_text=False)