#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import cv2
import random
import numpy as np
import rospy
from geometry_msgs.msg import Twist
import os
import pickle
import MySQLdb
import json
import math
import time

# initialize the current frame of the video, along with the list of ROI
# points along with whether or not this is input mode
frame = None
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MINI = 6

##**********MySQL********##
IP = '219.216.72.117'
PORT = 3306
USER_NAME = 'shuai'
PASSWD = 'shuai'
pretrain_DATABASE = 'PFR_DB'
TABLENAME = 'list_d'
##**********MySQL********##

GAME = 'follower'  # the name of the game being played for log files
ACTIONS = 3  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

#forward_speed = 0.35
forward_speed = 0.7
turn_speed = 0.3
portion_threshod = 0.06  # tested

action_dict = {1: np.array([forward_speed, 0, 0]), 0: np.array(
    [forward_speed, turn_speed, 0]), 2: np.array([forward_speed, 0, -turn_speed])}


def comeback(action_list, move_cmd, cmd_vel, r):

    # select the opposite action
    for i in range(len(action_list) - 1, -1, -1):
        pop_action = action_list.pop(i)
        current_at = action_dict[list(pop_action).index(1)]
        move_cmd.linear.x = -1 * current_at[0]
        move_cmd.angular.z = -1 * current_at[1] + -1 * current_at[2]
        print("move_cmd.linear.x is:{},move_cmd.angular.z is:{}".format(
            move_cmd.linear.x, move_cmd.angular.z))
        cmd_vel.publish(move_cmd)
        r.sleep()

    print("i have come back")

def array_to_json(*args):
    json_list=[]
    for item in args:
        item_list=item.tolist()
        item_json=json.dumps(item_list)
        json_list.append(item_json)
    return tuple(json_list)

def get_terminal_reward_and_writedata(s_t_orig, a_t, s_t1_orig, camera):
    global frame, data_num, action_list, D
    try:
        if rospy.is_shutdown() is True:
            # then person is too much far away from the center of frame
            raise KeyboardInterrupt
        else:
            terminal = False
            r_t = 1
            D.append((s_t_orig, a_t, r_t, s_t1_orig, terminal))
            if len(D) > REPLAY_MEMORY:
                D.pop(0)

    except KeyboardInterrupt:
        print("I can't see and i HAVE TO TERMINATE")
        terminal = True
        r_t = -10
        D.append((s_t_orig, a_t, r_t, s_t1_orig, terminal))
        with open("data/list_D.txt", "wb") as fp:
            pickle.dump(D, fp)
        print("has stored into list_D")
        os.system('echo %d > data/data_num.txt' % data_num)

        s_t_json, a_t_json, s_t1_json = array_to_json(s_t_orig, a_t, s_t1_orig)
        id, s_t, a_t, s_t1 = data_num, s_t_json, a_t_json, s_t1_json
        query = """INSERT INTO list_d VALUES (%s, %s, %s, %s, %s, %s)"""
        arg = (id, s_t, a_t, r_t, s_t1, terminal)
        print("data_num is:{},s_t_json len is:{},s_t1_json len is:{},r_t is:{},terminal is:{}".format(data_num, len(s_t), len(s_t1), r_t, terminal))
        cursor.execute(query, arg)
        db.commit()
        print("has committed to mysql")

        comeback(action_list, move_cmd, cmd_vel, r)
        shutdown()  # turtlebot shutdown
        camera.release()

    return r_t, terminal


def get_state(action, camera, move_cmd, cmd_vel, r):
    '''THIS FUNCTION is to return the next state s_t1 after taking action at state s_t'''
    global frame, data_num
    action_array = action_dict[list(action).index(1)]
    if action[1] == 0:
        #move_cmd.linear.x = 0.25
	move_cmd.linear.x = 0.5
        move_cmd.angular.z = action_array[1] + action_array[2]

    else:
        move_cmd.linear.x = action_array[0]
        move_cmd.angular.z = action_array[1] + action_array[2]

    cmd_vel.publish(move_cmd)
    r.sleep()
    print("linear_x&angular_z is:", move_cmd.linear.x, move_cmd.angular.z)

    grabbed, frame = camera.read()
    cv2.imwrite('data/saved_frames/' + np.str(data_num) + '.jpg', frame)
    if not grabbed:
        print("not grabbed frame")
    frame_hsv = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2HSV)
    print("frame_hsv shape is:", frame_hsv.shape)
    return frame_hsv


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


def createNetwork():
    with tf.name_scope('input_states'):
        s = tf.placeholder("float", [None, 60, 80, 12])

    # network weights
    with tf.name_scope('weights_and_bias'):
        W_conv1 = weight_variable([8, 8, 12, 32], "W_conv1")
        b_conv1 = bias_variable([32], "b_conv1")

        W_conv2 = weight_variable([4, 4, 32, 64], "W_conv2")
        b_conv2 = bias_variable([64], "b_conv2")

        W_conv3 = weight_variable([2, 2, 64, 64], "W_conv3")
        b_conv3 = bias_variable([64], "b_conv3")

        W_fc1 = weight_variable([384, 384], "W_fc1")
        b_fc1 = bias_variable([384], "b_fc1")

        W_fc2 = weight_variable([384, ACTIONS], "W_fc2")
        b_fc2 = bias_variable([ACTIONS], "b_fc2")


    # hidden layers
    with tf.name_scope('conv_layer_1'):
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 2) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv_layer_2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv_layer_3'):
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, 1) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('feature_repre_layer'):
        h_conv3_flat = tf.reshape(h_pool3, [-1, 384])

    with tf.name_scope('fcn_layer_1'):
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout_layer'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout layer
    with tf.name_scope('action_value_layer'):
        readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return s, readout, h_fc1, keep_prob


def shutdown():
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
    cmd_vel.publish(Twist())
    rospy.sleep(1)


def trainNetwork(s, readout, h_fc1, keep_prob, sess):
    # define the cost function

    # **********************initialize the camera*********************
    global frame
    camera = cv2.VideoCapture(0)
    #camera = cv2.VideoCapture('sample.mov')
    # **********************camera initialize end*********************

    # get the first state:doing nothing([0,1,0]) and preprocess the image
    # to 60*80x12
    do_nothing = np.zeros(ACTIONS)
    do_nothing[1] = 1

    x_t_orig = get_state(
        do_nothing, camera, move_cmd, cmd_vel, r)
    print("x_t_orig shape is:", x_t_orig.shape)
    s_t_orig = np.concatenate((x_t_orig, x_t_orig, x_t_orig, x_t_orig), axis=2)
    print("s_t_orig shape is:", s_t_orig.shape)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", "saved_networks/follower_dqn")
    else:
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON
    global data_num, action_list
    action_list = []
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t_orig], keep_prob: 1.0})[0]
        a_t = np.zeros([ACTIONS])
        if data_num % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(
                    ACTIONS)  # range(3) return:0/1/2
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1      # a_t: [1,0,0]  [0,1,0]  [0,0,1]
        else:
            a_t[1] = 1  # go forward  a_t:[0,1,0]
        action_list.append(a_t)

        # scale down epsilon
        if epsilon > FINAL_EPSILON and data_num > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_orig = get_state(
            a_t, camera, move_cmd, cmd_vel, r)

        s_t1_orig = np.append(s_t_orig[:, :, 3:12], x_t1_orig, axis=2)
        print("s_t1_orig shape is:", s_t1_orig.shape)

        r_t, terminal = get_terminal_reward_and_writedata(
            s_t_orig, a_t, s_t1_orig, camera)

        # update the old values
        s_t_orig = s_t1_orig
        data_num += 1

        # print info
        state = ""
        if data_num <= OBSERVE:
            state = "observe"
        elif data_num > OBSERVE and data_num <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", data_num, "/ STATE", state, "terminal", terminal,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))
        os.system('echo %d > data/data_num.txt' % data_num)
        os.system('echo %s >> data/max_q_value.txt' % np.max(readout_t))


if __name__ == "__main__":
    rospy.init_node('GoForward', anonymous=False)
    # tell user how to stop TurtleBot
    rospy.loginfo("To stop TurtleBot CTRL + C")
    # What function to call when you ctrl + c
    rospy.on_shutdown(shutdown)
    cmd_vel = rospy.Publisher(
        'cmd_vel_mux/input/navi', Twist, queue_size=10)
    r = rospy.Rate(20)
    move_cmd = Twist()

    if os.path.isfile("data/data_num.txt"):
        print("data_num has existed")
        data_num = os.popen('cat data/data_num.txt')
        data_num = int(data_num.read())
    else:
        data_num = 0
        os.system('echo %d > data/data_num.txt' % data_num)

    db = MySQLdb.connect(host=IP, port=PORT,
                         user=USER_NAME, passwd=PASSWD, db=pretrain_DATABASE)
    print("has connected successfully")
    cursor = db.cursor()

    if os.path.isfile("data/list_D.txt"):
        print("list_D has existed")
        with open("data/list_D.txt", "rb") as fp:
            D = pickle.load(fp)
    else:
        D = []

    sess = tf.InteractiveSession()
    s, readout, h_fc1, keep_prob = createNetwork()
    print("s, readout, h_fc1 is:", s, readout, h_fc1)
    trainNetwork(s, readout, h_fc1, keep_prob, sess)

