import tensorflow.compat.v1 as tf


graph = tf.get_default_graph()

sess = tf.Session()

saver = tf.train.import_meta_graph('./trained_models/tshirt/garment_fit/model.meta')
saver.restore(sess, 'model.data-00000-of-00001')

tf.train.write_graph(sess.graph_def, '.', 'graph.pb', as_text=False)