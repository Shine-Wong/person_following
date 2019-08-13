#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import random
import numpy as np
import MySQLdb
import json
np.set_printoptions(threshold=np.nan)

# list_cnn_2value: 1~1405:old_data; >1406:new data
# list_cnn_2_value_new_data:only new data


GAME = 'follower'  # the name of the game being played for log files
ACTIONS = 3  # number of valid actions
BATCH = 32  # size of minibatch



def test_next_batch(cursor, BATCH, start_id, end_id):
    '''
    start_id = 1001
    end_id = 1405
    '''
    batch_id =range(start_id, end_id + 1)  # [12,34,6,7,68,0]

    batch_tuple = tuple(batch_id)

    batch_query = """select s_t, a_t from list_cnn_3value where id in """
    cursor.execute(batch_query + str(batch_tuple))
    results = cursor.fetchall()
    s_j_batch = [np.array(json.loads(result[0])) for result in results]
    a_batch = [np.array(json.loads(result[1])) for result in results]
    return s_j_batch, a_batch
    

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def create_pretrain_Network():
    # network weights
    W_conv1 = weight_variable([8, 8, 12, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([2, 2, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([384, 384])
    b_fc1 = bias_variable([384])

    W_fc2 = weight_variable([384, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])


    # input layer
    s = tf.placeholder("float", [None, 60, 80, 12])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 2) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_conv3_flat = tf.reshape(h_pool3, [-1, 384])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # readout = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return s, readout, keep_prob, W_conv1
    
def predict(s, readout, keep_prob, W_conv1,sess, cursor):
    a = tf.placeholder("float", [None, ACTIONS])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=a, logits=readout))
    tf.scalar_summary('cross_entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(readout, 1), tf.argmax(a, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded: saved_networks/follower-dqn")
    else:
        print("Could not find old network weights")
    start_id=63
    end_id=81
    test_s_batch,test_a_batch=test_next_batch(cursor, BATCH, start_id, end_id)
    readout_batch=sess.run(readout,feed_dict={a:test_a_batch,s:test_s_batch,keep_prob:1.0})
    print("during training, batch_readout is {}".format(readout_batch))
    
if __name__ == "__main__":
    db = MySQLdb.connect(host='localhost', port=3306,
                         user='root', passwd='123', db='dd')
    print("has connected successfully")
    cursor = db.cursor()
    sess = tf.InteractiveSession()
    s, readout, keep_prob, W_conv1=create_pretrain_Network()
    predict(s, readout, keep_prob, W_conv1,sess, cursor)
