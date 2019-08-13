#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import random
import numpy as np
import pymysql
import json
import os
import pickle
from tensorflow.core.protobuf import saver_pb2
np.set_printoptions(threshold=np.nan)

# list_cnn_2value: 1~1405:old_data; >1406:new data
# list_cnn_2_value_new_data:only new data

IP = conf.IP
PORT = conf.PORT
USER_NAME = conf.USER_NAME
PASSWD = conf.PASSWD
pretrain_DATABASE = conf.pretrain_DATABASE
DQN_TABLENAME = conf.DQN_TABLENAME


GAME = 'follower'  # the name of the game being played for log files
ACTIONS = 3  # number of valid actions
BATCH = 32  # size of minibatch
MAX_BATCH = 1
GAMMA = 0.99


def train_next_batch(cursor, BATCH):

    #max_id_query = """select max(id) from list_dd_and_color"""
    max_id_query = """select max(id) from　"""+DQN_TABLENAME
    cursor.execute(max_id_query)
    max_id = cursor.fetchall()  # a tuple ((max_id))
    max_id = max_id[0][0]
    random_batch_id = random.sample(
        range(0, max_id + 1), BATCH)  # [12,34,6,7,68,0]

    random_batch_tuple = tuple(random_batch_id)

    batch_query = """select s_t_orig, a_t, r_t, s_t1_orig, terminal from list_dd_and_color where id in """
    cursor.execute(batch_query + str(random_batch_tuple))
    results = cursor.fetchall()
    s_j_batch = [np.array(json.loads(result[0])) for result in results]
    a_batch = [np.array(json.loads(result[1])) for result in results]
    r_batch = [result[2] for result in results]
    s_j1_batch = [np.array(json.loads(result[3])) for result in results]
    terminal_batch = [result[4] for result in results]
    return s_j_batch, a_batch, r_batch, s_j1_batch, terminal_batch


def test_next_batch(cursor, BATCH):
    '''
    start_id = 1001
    end_id = 1405
    '''
    start_id = 2992
    end_id = 4992
    random_batch_id = random.sample(
        range(start_id, end_id + 1), BATCH)  # [12,34,6,7,68,0]

    random_batch_tuple = tuple(random_batch_id)

    batch_query = """select s_t, a_t from new_cnn_3value where id in """
    cursor.execute(batch_query + str(random_batch_tuple))
    results = cursor.fetchall()
    s_j_batch = [np.array(json.loads(result[0])) for result in results]
    a_batch = [np.array(json.loads(result[1])) for result in results]
    return s_j_batch, a_batch


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary('histogram/' + name, var)


def create_pretrain_Network():
    # network weights
    with open("weights_of_3channel_CNN.txt", "rb") as fp:
        W_conv1_cnn = pickle.load(fp)
        b_conv1_cnn = pickle.load(fp)
        W_conv2_cnn = pickle.load(fp)
        b_conv2_cnn = pickle.load(fp)
        W_conv3_cnn = pickle.load(fp)
        b_conv3_cnn = pickle.load(fp)
        W_fc1_cnn = pickle.load(fp)
        b_fc1_cnn = pickle.load(fp)
        W_fc2_cnn = pickle.load(fp)
        b_fc2_cnn = pickle.load(fp)
    W_conv1 = tf.Variable(W_conv1_cnn, dtype=tf.float32)
    b_conv1 = tf.Variable(b_conv1_cnn, dtype=tf.float32)

    W_conv2 = tf.Variable(W_conv2_cnn, dtype=tf.float32)
    b_conv2 = tf.Variable(b_conv2_cnn, dtype=tf.float32)

    W_conv3 = tf.Variable(W_conv3_cnn, dtype=tf.float32)
    b_conv3 = tf.Variable(b_conv3_cnn, dtype=tf.float32)

    W_fc1 = tf.Variable(W_fc1_cnn, dtype=tf.float32)
    b_fc1 = tf.Variable(b_fc1_cnn, dtype=tf.float32)

    W_fc2 = tf.Variable(W_fc2_cnn, dtype=tf.float32)
    b_fc2 = tf.Variable(b_fc2_cnn, dtype=tf.float32)

    '''
    W_conv1 = weight_variable([8, 8, 4, 32], "W_conv1")
    b_conv1 = bias_variable([32], "b_conv1")

    W_conv2 = weight_variable([4, 4, 32, 64], "W_conv2")
    b_conv2 = bias_variable([64], "b_conv2")

    W_conv3 = weight_variable([2, 2, 64, 64], "W_conv3")
    b_conv3 = bias_variable([64], "b_conv3")

    W_fc1 = weight_variable([384, 384], "W_fc1")
    b_fc1 = bias_variable([384], "b_fc1")

    W_fc2 = weight_variable([384, ACTIONS], "W_fc2")
    b_fc2 = bias_variable([ACTIONS], "b_fc2")
    '''
    variable_summaries(W_conv1, 'W_conv1')
    variable_summaries(b_conv1, 'b_conv1')
    variable_summaries(W_conv2, 'W_conv2')
    variable_summaries(b_conv2, 'b_conv2')
    variable_summaries(W_conv3, 'W_conv3')
    variable_summaries(b_conv3, 'b_conv3')
    variable_summaries(W_fc1, 'W_fc1')
    variable_summaries(b_fc1, 'b_fc1')
    variable_summaries(W_fc2, 'W_fc2')
    variable_summaries(b_fc2, 'b_fc2')

    # input layer
    s = tf.placeholder("float", [None, 60, 80, 12])
    #s = tf.placeholder(tf.float32, [None, 60, 80, 12])

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
    #keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # readout = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return s, readout, keep_prob, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2


def train_and_test_cnn_Network(s, readout, keep_prob, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, sess, cursor):
        # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder(tf.float32, [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #saver = tf.train.Saver()
    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1) 
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(
        'summary_test/using_pretrained_model/train', sess.graph)
    # test_writer = tf.train.SummaryWriter(
    #   'summary/using_pretrained_model/test', sess.graph)

    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded: saved_networks/follower_dqn")
    else:
        print("Could not find old network weights")

    # *****************train_process_start*******************/

    for i in range(MAX_BATCH):
        s_j_batch, a_batch, r_batch, s_j1_batch, terminal_batch = train_next_batch(
            cursor, BATCH)

        y_batch = []
        readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch,keep_prob:0.5})
        sum_max_Q = 0
        for j in range(0, BATCH):
            terminal = terminal_batch[j]
            if terminal == 1:
                y_batch.append(r_batch[j])
            else:
                y_batch.append(r_batch[j] + GAMMA *
                               np.max(readout_j1_batch[j]))
            sum_max_Q += np.max(readout_j1_batch[j])
        average_max_Q = sum_max_Q / float(BATCH)

        _, c, summ_train = sess.run([train_step, cost, merged], feed_dict={
            y: y_batch,
            a: a_batch,
            s: s_j_batch,
            keep_prob: 0.5})
        print("during training, STEP is {},cost is {},average max Q is{}".format(
            i, c, average_max_Q))
        os.system('(echo average_max_Q is:%f) >> average_max_Q.txt' %
                  average_max_Q)  # write into file start with a new line

        train_writer.add_summary(summ_train, i)
        # *********************train_process_end**********************/

    # *****************test_process_start*******************/
    '''
    for i in range(100):
        test_s_j_batch, test_a_batch = test_next_batch(cursor, BATCH)
        train_accuracy, summ_test = sess.run([accuracy, merged], feed_dict={
            a: test_a_batch,
            s: test_s_j_batch,
            keep_prob: 1.0})
        print("during testing, accuracy is {}".format(train_accuracy))
        test_writer.add_summary(summ_test, i)
    '''
    a = saver.save(sess, 'saved_networks/follower_dqn')
    # print("a is:",a)   #saved_networks/follower_dqn
    if a == 'saved_networks/follower_dqn':
        print(
            'model has been saved successfully.')


def playGame():
    sess = tf.InteractiveSession()
    s, readout, keep_prob, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2 = create_pretrain_Network()
    train_and_test_cnn_Network(s, readout, keep_prob, W_conv1, b_conv1, W_conv2,
                               b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, sess, cursor)


if __name__ == "__main__":
    '''
    if tf.gfile.Exists('summary'):
        tf.gfile.DeleteRecursively('summary')
    tf.gfile.MakeDirs('summary')
    '''
    db = pymysql.connect(host=IP, port=PORT, user=USER_NAME, passwd=PASSWD, db=pretrain_DATABASE)
    print("has connected successfully")
    cursor = db.cursor()
    playGame()
