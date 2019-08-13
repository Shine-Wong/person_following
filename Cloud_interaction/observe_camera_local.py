#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import cv2
import random
import numpy as np
import rospy
from geometry_msgs.msg import Twist
import os
import json
import MySQLdb
import math
import time

# initialize the current frame of the video, along with the list of ROI
# points along with whether or not this is input mode
frame = None
roiPts = []
inputMode = False
roiBox = None
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MINI = 6

IP = 'localhost'
PORT = 3306
USER_NAME = 'root'
PASSWD = '123'
pretrain_DATABASE = 'dd'
#TABLENAME = 'list_dd_and_color'
TABLENAME = 'cnn_based_dqn'

GAME = 'follower'  # the name of the game being played for log files
ACTIONS = 3  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

lower_shang = np.array([20, 100, 90])
upper_shang = np.array([30, 255, 255])

lower_xia = np.array([0, 0, 0])
upper_xia = np.array([180, 255, 30])

forward_speed = 0.1
turn_speed = 0.12
portion_threshod = 0.06  # tested

action_dict = {0: np.array([forward_speed, 0, 0]), 1: np.array(
    [forward_speed, turn_speed, 0]), 2: np.array([forward_speed, 0, -turn_speed])}


def resize_and_get_mask(image):
    frm_hsv = cv2.cvtColor(cv2.resize(image, (80, 60)), cv2.COLOR_BGR2HSV)
    mask_shang = cv2.inRange(frm_hsv, lower_shang, upper_shang)
    mask_xia = cv2.inRange(frm_hsv, lower_xia, upper_xia)
    mask_full = mask_shang + mask_xia  # 1-channel image
    return mask_full


def person_position_and_nonzero_portion(image=frame):
    mask_full = resize_and_get_mask(frame)  # 1-channel image
    rows, cols = mask_full.shape
    nonzeroindex = np.nonzero(mask_full)
    # high_mean = np.mean(nonzeroind[0])  # mean index of person in height
    wide_mean = np.mean(nonzeroindex[1])  # mean index of person in width
    nonzero_num = (mask_full != 0).sum()
    nonzero_portion = float(nonzero_num) / (cols * rows)
    return wide_mean, nonzero_portion


def selectROI(event, x, y, flags, param):
    # grab the reference to the current frame, list of ROI
    # points and whether or not it is ROI selection mode
    global frame, roiPts, inputMode, roiHist

    # if we are in ROI selection mode, the mouse was clicked,
    # and we do not already have four points, then update the
    # list of ROI points with the (x, y) location of the click
    # and draw the circle
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def first_ROI_reward():
    global frame, roiPts, roiBox, roiHist, inputMode
    # roiBox, roiHist = determine_ROI_for_first_time()
    # indicate that we are in input mode and clone the frame
    inputMode = True
    orig = frame.copy()
    while len(roiPts) < 4:
        cv2.imshow("frame", frame)
        cv2.waitKey(0)
    roiPts = np.array(roiPts)
    s = roiPts.sum(axis=1)
    tl = roiPts[np.argmin(s)]
    br = roiPts[np.argmax(s)]
    # grab the ROI for the bounding box and convert it
    # to the HSV color space
    roi = orig[tl[1]:br[1], tl[0]:br[0]]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # compute a HSV histogram for the ROI and store the
    # bounding box
    roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
    roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
    roiBox = (tl[0], tl[1], br[0], br[1])
    reward = 0.5
    return roiBox, roiHist, reward


def calculate_reward_using_camshift(termination, eposilon):
    global frame, roiBox, roiHist
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
    (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
    pts = np.int0(cv2.cv.BoxPoints(r))
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    roi_mid = 0.5 * (pts[0][0] + pts[3][0])  # middle of the ROI
    frame_mid = 0.5 * FRAME_WIDTH  # middle of the frame
    devirance = abs(frame_mid - roi_mid)
    reward = eposilon - devirance
    return reward


def calculate_reward_using_portion():
    global frame
    position, portion = person_position_and_nonzero_portion(image=frame)
    if portion < portion_threshod:
        reward = -1
    else:
        if position >= 40:
            reward = -1 * position / 20 + 3
        if position < 40:
            reward = position / 20 - 1
    return reward


def go_forward(move_cmd, cmd_vel, r):
    current_at = action_dict[0]
    move_cmd.linear.x = current_at[0]
    move_cmd.angular.z = current_at[1] + current_at[2]
    cmd_vel.publish(move_cmd)
    r.sleep()
    print("go forward")


def turn_left(move_cmd, cmd_vel, r):
    current_at = action_dict[1]
    move_cmd.linear.x = current_at[0]
    move_cmd.angular.z = current_at[1] + current_at[2]
    cmd_vel.publish(move_cmd)
    r.sleep()
    print("turn left")


def turn_right(move_cmd, cmd_vel, r):
    current_at = action_dict[2]
    move_cmd.linear.x = current_at[0]
    move_cmd.angular.z = current_at[1] + current_at[2]
    cmd_vel.publish(move_cmd)
    r.sleep()
    print("turn left")


def comeback(action_list, move_cmd, cmd_vel, r):
    '''
    move_cmd.linear.x = 0
    move_cmd.angular.z = math.radians(45)  # 45 degree/s
    for i in range(40):  # 0.1*40*45=180 degree
        # turn back
        cmd_vel.publish(move_cmd)
        r.sleep()
    # go forward
    move_cmd.linear.x = forward_speed
    move_cmd.angular.z = 0
    cmd_vel.publish(move_cmd)
    r.sleep()
    '''
    # select the opposite action
    for i in range(len(action_list) - 1, -1, -1):
        pop_action = action_list.pop(i)
        current_at = action_dict[list(pop_action).index(1)]
        move_cmd.linear.x = -3*current_at[0]
        move_cmd.angular.z = -3*current_at[1] + -1*current_at[2]
        print("move_cmd.linear.x is:{},move_cmd.angular.z is:{}".format(move_cmd.linear.x, move_cmd.angular.z))
        cmd_vel.publish(move_cmd)
        r.sleep()
        
    print("i have come back")
    '''
        if pop_action == np.array([1, 0, 0]):
            go_forward(move_cmd, cmd_vel, r)
        if pop_action == np.array([0, 1, 0]):
            turn_left(move_cmd, cmd_vel, r)
        if pop_action == np.array([0, 0, 1]):
            turn_right(move_cmd, cmd_vel, r)
    '''

def array_to_json(*args):
    json_list=[]
    for item in args:
        item_list=item.tolist()
        item_json=json.dumps(item_list)
        json_list.append(item_json)
    return tuple(json_list)


def get_terminal_and_writedata(s_t, a_t, r_t, s_t1, s_t_orig, s_t1_orig, camera):
    global frame, data_num, action_list
    try:
        person_position, person_portion = person_position_and_nonzero_portion(
            image=frame)
        print("position of person is:", person_position)
        print("portion of person is:", person_portion)
        terminal = False
        if person_portion < portion_threshod:
            # then person is too much far way from the center of frame
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("I can't see and i HAVE TO TERMINATE")
        terminal = True
        camera.release()
        cv2.destroyAllWindows()
        comeback(action_list, move_cmd, cmd_vel, r)
        shutdown()  # turtlebot shutdown
    finally:
        s_t_json,a_t_json,s_t1_json,s_t_orig_json,s_t1_orig_json=array_to_json(s_t,a_t,s_t1,s_t_orig,s_t1_orig)
        query = """INSERT INTO cnn_based_dqn VALUES (%s, %s,%s, %s, %s, %s, %s, %s)"""
        arg = (data_num, s_t_json, a_t_json, r_t, s_t1_json,
               terminal, s_t_orig_json, s_t1_orig_json)
        print("data_num is:{},s_t_json len is:{},s_t1_json len is:{},r_t is:{},terminal is:{}, s_t_orig_json len is:{}, s_t1_orig_json len is:{}".format(
            data_num, len(s_t_json), len(s_t1_json), r_t, terminal, len(s_t_orig_json), len(s_t1_orig_json)))
        cursor.execute(query, arg)
        db.commit()
        print("has committed to mysql")
        os.system('echo %d > data_num.txt' % data_num)
    return terminal


def get_state(action, camera, move_cmd, cmd_vel, r, termination, eposilon=14):
    global roiPts, roiBox, frame, inputMode, roiHist, data_num
    action_array = action_dict[list(action).index(1)]
    if action[0]==0:
        move_cmd.linear.x=0.07
        move_cmd.angular.z = action_array[1] + action_array[2]

    else:
        move_cmd.linear.x = action_array[0]
        move_cmd.angular.z = action_array[1] + action_array[2]
    
    cmd_vel.publish(move_cmd)
    r.sleep()
    print("linear_x&angular_z is:", move_cmd.linear.x, move_cmd.angular.z)

    grabbed, frame = camera.read()
    cv2.imwrite('saved_frames/' + np.str(data_num) + '.jpg', frame)
    if not grabbed:
        print("not grabbed frame")
    if roiBox is not None:
        #reward = calculate_reward_using_camshift(termination, eposilon)
        reward = calculate_reward_using_portion()
    # select ROI for the first time
    key = cv2.waitKey(100) & 0xFF
    if roiBox is None:
        roiBox, roiHist, reward = first_ROI_reward()
    print(roiBox)

    return frame, reward


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
    # network weights
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


    # input layer
    s = tf.placeholder("float", [None, 60, 80, 4])

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
    return s, readout, h_fc1, keep_prob


def shutdown():
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
    cmd_vel.publish(Twist())
    rospy.sleep(1)


def trainNetwork(s, readout, h_fc1, keep_prob, sess):
    # define the cost functio

    # **********************initialize the camera*********************
    global frame, roiPts, inputMode, roiHist
    camera = cv2.VideoCapture(1)
    #camera = cv2.VideoCapture('sample.mov')
    # **********************camera initialize end*********************
    # setup the mouse callback
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # get the first state:doing nothing([1,0,0]) and preprocess the image
    # to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    x_t_color, r_0 = get_state(
        do_nothing, camera, move_cmd, cmd_vel, r, termination, eposilon=14)

    x_t_orig = cv2.cvtColor(cv2.resize(x_t_color, (80, 60)), cv2.COLOR_BGR2HSV)
    print("x_t_orig shape is:", x_t_orig.shape)
    s_t_orig = np.concatenate((x_t_orig, x_t_orig, x_t_orig, x_t_orig), axis=2)
    print("s_t_orig shape is:", s_t_orig.shape)

    x_t = resize_and_get_mask(x_t_color)
    print("x_t shape is:", x_t.shape)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    print("s_t shape is:", s_t.shape)

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
            # ir runs much slower if add the following clauses

        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t],keep_prob:1.0})[0]
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
            a_t[0] = 1  # go forward  a_t:[1,0,0]
        action_list.append(a_t)

        # scale down epsilon
        if epsilon > FINAL_EPSILON and data_num > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_color, r_t = get_state(
            a_t, camera, move_cmd, cmd_vel, r, termination, eposilon=14)

        x_t1_orig = cv2.cvtColor(cv2.resize(
            x_t1_color, (80, 60)), cv2.COLOR_BGR2HSV)
        print("x_t1_orig shape is:", x_t1_orig.shape)
        s_t1_orig = np.append(x_t1_orig, s_t_orig[:, :, :9], axis=2)
        print("s_t1_orig shape is:", s_t1_orig.shape)

        x_t1 = resize_and_get_mask(x_t1_color)
        x_t1 = np.reshape(x_t1, (60, 80, 1))
        print("x_t1 shape is:", x_t1.shape)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        print("S_t1[0] shape is:", s_t1[0].shape)

        terminal = get_terminal_and_writedata(
            s_t, a_t, r_t, s_t1, s_t_orig, s_t1_orig, camera)
        # update the old values
        s_t = s_t1
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
        os.system('echo %d > data_num.txt' % data_num)


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

    data_num = 0
    data_num = os.popen('cat data_num.txt')
    data_num = int(data_num.read())
    db = MySQLdb.connect(host=IP, port=PORT,
                         user=USER_NAME, passwd=PASSWD, db=pretrain_DATABASE)

    print("has connected successfully")
    cursor = db.cursor()

    sess = tf.InteractiveSession()
    s, readout, h_fc1, keep_prob = createNetwork()
    print("s, readout, h_fc1 is:", s, readout, h_fc1)
    trainNetwork(s, readout, h_fc1, keep_prob, sess)
