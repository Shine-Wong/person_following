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


# initialize the current frame of the video, along with the list of ROI
# points along with whether or not this is input mode
frame = None
roiPts = []
inputMode = False
roiBox = None
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MINI=6
lower_yellow=np.array([20,100,100])
upper_yellow=np.array([30,255,255])
    
lower_black=np.array([0,0,0])
upper_black=np.array([180,255,30])
    
    

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

# we can send a parameter terminal to get_state,the parameter is defined out of this function
# is is defined in the train_network_function,for example,the rospy.shutdown()


def get_state(action, camera, move_cmd,cmd_vel,r, termination, eposilon=14):
    global roiPts, roiBox, frame, inputMode, roiHist, data_num
    x_z = np.array([0.2, 0.2, -0.2])
    x_z_value = action * x_z
    move_cmd.linear.x = x_z_value[0]
    move_cmd.angular.z = x_z_value[1] + x_z_value[2]
    cmd_vel.publish(move_cmd)
    r.sleep()
    print("linear_x&angular_z is:", move_cmd.linear.x, move_cmd.angular.z)
    grabbed, frame = camera.read()
    cv2.imwrite('saved_frames/'+np.str(data_num)+'.jpg',frame)
    if not grabbed:
        print("not grabbed frame")
    #print("roiBox is:", roiBox)
    if roiBox is not None:
        #reward = do_camshift(frame, roiBox, roiHist, termination, eposilon=1)
        # convert the current frame to the HSV color space
        # and perform mean shift
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
        (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
        pts = np.int0(cv2.cv.BoxPoints(r))
        #print("pts is:", pts)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        roi_mid = 0.5 * (pts[0][0] + pts[3][0])  # middle of the ROI
        frame_mid = 0.5 * FRAME_WIDTH  # middle of the frame
        devirance = abs(frame_mid - roi_mid)
        if devirance < eposilon:
            reward = 1          # reward = 1  return : r_t
        else:
            reward = -1
     # select ROI for the first time
    key = cv2.waitKey(100) & 0xFF
    # if key == ord("i") and len(roiPts) < 4:
    if roiBox is None:
        # roiBox, roiHist = determine_ROI_for_first_time()
        # indicate that we are in input mode and clone the frame
        print(key == ord("i") and len(roiPts) < 4)
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

        reward = 0
    print(roiBox)
    terminal = False
    print("right now, rospy.is_shutdown is :", rospy.is_shutdown())
    if rospy.is_shutdown() is True:
        # if reward is not None and rospy.is_shutdown():
        terminal = True
    # if key == ord("q"):
        camera.release()
        cv2.destroyAllWindows()
    return frame, reward, terminal


GAME = 'follower'  # the name of the game being played for log files
ACTIONS = 3  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1


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


def shutdown():
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
    # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop
    # TurtleBot
    cmd_vel.publish(Twist())
    # sleep just makes sure TurtleBot receives the stop command prior to
    # shutting down the script
    rospy.sleep(1)


def trainNetwork(s, readout, h_fc1, sess):
    # define the cost functio

    # **********************initialize the camera*********************
    global frame, roiPts, inputMode, roiHist
    camera = cv2.VideoCapture(1)
    #camera = cv2.VideoCapture(
    #    'sample.mov')
    # **********************camera initialize end*********************
    # setup the mouse callback
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", selectROI)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # printing
    # a_file = open("logs_" + GAME + "/readout.txt", 'w')
    # h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state:doing nothing([1,0,0]) and preprocess the image
    # to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1

    x_t, r_0, terminal = get_state(do_nothing, camera, move_cmd,cmd_vel,r, termination, eposilon=14)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 60)), cv2.COLOR_BGR2HSV)
    mask_yellow=cv2.inRange(x_t,lower_yellow, upper_yellow)
    mask_black=cv2.inRange(x_t,lower_black, upper_black)
    x_t=mask_yellow+mask_black
    #print("x_t shape is:", x_t.shape)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print("s_t shape is:", s_t.shape)

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
    global data_num
    while "flappy bird" != "angry bird":
            # ir runs much slower if add the following clauses

        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if data_num % FRAME_PER_ACTION == 0:
            a_t[0] = 1   # keep the forward velocity
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(
                    1, ACTIONS)  # range(1,3) return:1 or 2
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1      # a_t:[1,1,0] or [1,0,1]
        else:
            a_t[0] = 1  # go forward  a_t:[1,0,0]

        # scale down epsilon
        if epsilon > FINAL_EPSILON and data_num > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = get_state(
            a_t, camera, move_cmd,cmd_vel,r, termination, eposilon=14)
        x_t1 = cv2.cvtColor(cv2.resize(
            x_t1_colored, (80, 60)), cv2.COLOR_BGR2HSV)
        mask1_yellow=cv2.inRange(x_t1,lower_yellow, upper_yellow)
        mask1_black=cv2.inRange(x_t1,lower_black, upper_black)
        x_t1=mask1_yellow+mask1_black
        x_t1 = np.reshape(x_t1, (60, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        #print("s_t, a_t, r_t, s_t1, terminal is:",
         #     s_t, a_t, r_t, s_t1, terminal)

        #print("s_t type is:", type(s_t))

        # store the transition in D
        #D.append((s_t, a_t, r_t, s_t1, terminal))

        s_t_list = s_t.tolist()
        a_t_list = a_t.tolist()
        s_t1_list = s_t1.tolist()
        s_t_json = json.dumps(s_t_list)
        #print("a_t is:", a_t)
        #print("a_t_list length is:", len(a_t_list))
        #print("s_t_list length is:", len(s_t_list))
        #print("s_t_json is :", type(s_t_json), len(s_t_json))  # str 140960
        s_t_restore = json.loads(s_t_json)
        #print("s_t_restore is :", type(s_t_restore),
        #      len(s_t_restore))  # list, 80
        a_t_json = json.dumps(a_t_list)
        s_t1_json = json.dumps(s_t1_list)
        #print("s_t json len is:", len(s_t_json))
        #print("data_num_current type  is:", type(data_num))
        query = """INSERT INTO list_color VALUES (%s, %s,%s, %s, %s, %s)"""
        arg = (data_num, s_t_json, a_t_json, r_t, s_t1_json, terminal)
        cursor.execute(query, arg)
        db.commit()
        print("has commited to mysql")

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
    r = rospy.Rate(10)
    move_cmd = Twist()

    data_num = 0
    data_num = os.popen('cat data_num.txt')
    data_num = int(data_num.read())
    #db = MySQLdb.connect(host='localhost', port=3306,
     #                    user='root', passwd='123', db='dd')
    db = MySQLdb.connect(host='219.216.87.170', port=3306,
                         user='root', passwd='123', db='dd')
    print("has connected successfully")
    cursor = db.cursor()

    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    print("s, readout, h_fc1 is:", s, readout, h_fc1)
    trainNetwork(s, readout, h_fc1, sess)

