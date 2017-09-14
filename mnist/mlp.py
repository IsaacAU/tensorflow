import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.examples.tutorials.mnist import input_data

data_dir = "/tmp/mnist/mlp/data"
log_dir = "/tmp/mnist/mlp/log"
model_dir = "/tmp/mnist/mlp/model"

INPUT_DIM = 784
N_CLASSES = 10

tf.logging.set_verbosity(tf.logging.INFO)
builder = tf.saved_model.builder.SavedModelBuilder(model_dir)

def weights(shape):
    return tf.Variable(tf.truncated_normal(stddev=0.1, shape=shape), name="weights")

def biases(shape):
    return tf.Variable(tf.constant(0.1, shape=[shape], dtype=tf.float32), name="biases")

def weights_and_biases(shape):
    return weights(shape), biases(shape[1])

def infer(features, keep_prob):
    with tf.name_scope("layer1"):
        weights, biases = weights_and_biases([INPUT_DIM, 500])
        y1 = tf.nn.relu(tf.matmul(features, weights) + biases)
    with tf.name_scope("dropout1"):
        drop1 = tf.nn.dropout(y1, keep_prob)
    with tf.name_scope("layer2"):
        weights, biases = weights_and_biases([500, 100])
        y2 = tf.nn.relu(tf.matmul(drop1, weights) + biases)
    with tf.name_scope("dropout2"):
        drop2 = tf.nn.dropout(y2, keep_prob)
    with tf.name_scope("layer3"):
        weights, biases = weights_and_biases([100, N_CLASSES])
        logits = tf.matmul(drop2, weights) + biases
    return logits

def loss(logits, labels):
    x_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    x_entropy = tf.reduce_mean(x_entropy)
    return x_entropy

def digit_accuracy(accurate_mask, one_hot_labels, digit):
    positive_mask = tf.equal(one_hot_labels, digit)
    and_mask = tf.logical_and(accurate_mask, positive_mask)
    and_counts = tf.reduce_sum(tf.cast(and_mask, tf.int32))
    tf.summary.scalar("digit{}_accurate_counts".format(digit), and_counts)
    accu_rate = tf.reduce_mean(tf.cast(and_mask, tf.float32))
    tf.summary.scalar("digit{}_accurate_rate".format(digit), accu_rate)

def accuracy(logits, labels):
    accurate_counts = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accurate_sum = tf.reduce_sum(tf.cast(accurate_counts, tf.int32))
    tf.summary.scalar("accurate_counts", accurate_sum)
    for i in range(10):
        digit_accuracy(accurate_counts, tf.argmax(labels, 1), i)
    accurate_counts = tf.cast(accurate_counts, tf.float32)
    accurate_rate = tf.reduce_mean(accurate_counts)
    return accurate_rate

if __name__ == "__main__":
    sess = tf.Session()
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    with tf.name_scope("placeholder"):
        feature_ph = tf.placeholder(tf.float32, [None, INPUT_DIM], name="features")
        label_ph = tf.placeholder(tf.int32, [None, N_CLASSES], name="labels")
        keep_prob = tf.placeholder(tf.float32, shape=(), name="dropout_prob")
    logits = infer(feature_ph, keep_prob)
    cross_entropy = loss(logits, label_ph)
    tf.summary.scalar("cross_entropy", cross_entropy)
    accu = accuracy(logits, label_ph)
    tf.summary.scalar("accuracy", accu)
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
    merged = tf.summary.merge_all()    
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.tag_constants.TRAINING],
                                         signature_def_map=foo_signatures,
                                         assets_collection=foo_assets)
    builder.add_meta_graph([tf.saved_model.tag_constants.SERVING])
    tf.global_variables_initializer().run(session=sess)
    for i in range(1000):
        x, y = mnist.train.next_batch(100)
        summary, _ = sess.run([merged, train], feed_dict={feature_ph: x,
                                                          label_ph: y,
                                                          keep_prob: 0.5})
        train_writer.add_summary(summary, i)
        summary = sess.run(merged, feed_dict={feature_ph: mnist.test.images,
                                              label_ph: mnist.test.labels,
                                              keep_prob: 1.0})
        test_writer.add_summary(summary, i)
        print("{}th iteration..".format(i))

    train_writer.close()
    test_writer.close()
        
    
