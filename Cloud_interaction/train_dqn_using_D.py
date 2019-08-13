#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import random
import numpy as np
import MySQLdb
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)

IP = '219.216.72.117'
PORT=3306
USER_NAME= 'shuai'
PASSWD= 'shuai'
pretrain_DATABASE='PFR_DB'
TABLENAME='list_d'

GAME = 'follower'  # the name of the game being played for log files
ACTIONS = 3  # number of valid actions
GAMMA = 0.9  # decay rate of past observations
OBSERVE = 10000  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatchs
MAX_BATCH = 2000


def weight_variable(shape,):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial,)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1280, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 60, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    # h_pool3 = max_pool_2x2(h_conv3)

    # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1280])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    # only train if done observing
    # *****************train_process_start*******************/
    for i in range(MAX_BATCH):
        print("i is:", i)
        # sample a minibatch to train on
        # minibatch = random.sample(list_D, BATCH)
        max_id_query = """select max(id) from list_d"""
        cursor.execute(max_id_query)
        max_id = cursor.fetchall()  # a tuple ((max_id))
        max_id = max_id[0][0]

        random_batch_id = random.sample(
            range(max_id + 1), BATCH)  # [12,34,6,7,68,0]

        random_batch_tuple = tuple(random_batch_id)

        batch_query = """select s_t, a_t, r_t, s_t1, terminal from list_d where id in """
        cursor.execute(batch_query + str(random_batch_tuple))
        results = cursor.fetchall()
        # a length-BATCH list,each item is (80,80,4) array
        s_j_batch = [np.array(json.loads(result[0])) for result in results]
        a_batch = [np.array(json.loads(result[1])) for result in results]
        r_batch = [result[2] for result in results]
        s_j1_batch = [np.array(json.loads(result[3])) for result in results]

        y_batch = []
        readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
        for j in range(0, len(results)):
            terminal = results[j][4]
            # if terminal, only equals reward
            if terminal == 1:
                y_batch.append(r_batch[j])
            else:
                # np.max(readout_j1_batch[i]):the largest Q-value among all
                # actions for the current states
                y_batch.append(r_batch[j] + GAMMA *
                               np.max(readout_j1_batch[j]))

       # perform gradient step
        _, c = sess.run([train_step, cost], feed_dict={
            y: y_batch,
            a: a_batch,
            s: s_j_batch})

        print("STEP is {},cost is {}".format(i, c))
        # *********************train_process_end**********************/

    saver.save(sess, 'saved_networks/' + GAME + '-dqn')


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    print("s, readout, h_fc1 is:", s, readout, h_fc1)
    trainNetwork(s, readout, h_fc1, sess)


if __name__ == "__main__":
    db = MySQLdb.connect(host=IP, port=PORT,
                         user=USER_NAME, passwd=PASSWD, db=pretrain_DATABASE)
    print("has connected successfully")
    cursor = db.cursor()
    playGame()
