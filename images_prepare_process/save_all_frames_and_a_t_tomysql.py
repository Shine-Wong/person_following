import cv2
import os
import numpy as np
import json
import pymysql
import pickle
import conf


IP = conf.IP
PORT = conf.PORT
USER_NAME = conf.USER_NAME
PASSWD = conf.PASSWD
pretrain_DATABASE = conf.pretrain_DATABASE
TABLENAME = conf.TABLENAME

db = pymysql.connect(host=IP, port=PORT, user=USER_NAME,
                     passwd=PASSWD, db=pretrain_DATABASE)

total_sequences_num = conf.sequences_num  #(s_t,a_t) sequences
total_backgrounds = conf.after_luminance_backgrounds_num


def resize_and_tohsv(image):
    frm_hsv = cv2.cvtColor(cv2.resize(image, (80, 60)), cv2.COLOR_BGR2HSV)
    return frm_hsv


def action_label(db, root_dir, cnn_frame_num, background_index):

    # 0~275   483~565  700~795  806~849  854~953  1006~1139  1286~1305
    # 1313~1339   1389~1536  1838~1927  1963~2047  2055~2115  2197~ 2235
    # 2243~2329  2419, 2472  2480, 2585
    for i in range(0, total_sequences_num):
        
        frame1 = cv2.imread(os.path.join(
            root_dir, str(i) + '_' + str(background_index) + '.jpg'))
        frame2 = cv2.imread(os.path.join(root_dir, str(
            i + 1) + '_' + str(background_index) + '.jpg'))
        frame3 = cv2.imread(os.path.join(root_dir, str(
            i + 2) + '_' + str(background_index) + '.jpg'))
        frame4 = cv2.imread(os.path.join(root_dir, str(
            i + 3) + '_' + str(background_index) + '.jpg'))
        print(frame1.shape,frame2.shape,frame3.shape,frame4.shape)
       
        hsv_1 = resize_and_tohsv(frame1)
        hsv_2 = resize_and_tohsv(frame2)
        hsv_3 = resize_and_tohsv(frame3)
        hsv_4 = resize_and_tohsv(frame4)

        s_t = np.concatenate((hsv_1, hsv_2,
                              hsv_3, hsv_4), axis=2)

        print(hsv_1.shape)
        print(s_t.shape)

        a_t = list_a_t[i]
        a_t_json = json.dumps(a_t.tolist())
        s_t_json = json.dumps(s_t.tolist())
        cnn_frame_num = cnn_frame_num + 1

        #query = """INSERT INTO new_cnn_3value VALUESTABLE_NAME (%s, %s, %s)"""
        query = """INSERT INTO """+TABLENAME+""" (id,s_t,a_t) VALUES (%s, %s, %s)"""
        arg = (cnn_frame_num, s_t_json, a_t_json)
        cursor = db.cursor()
        cursor.execute(query, arg)
        db.commit()
        print("has committed to mysql")
        os.system('echo %d > cnn_frame_num.txt' % cnn_frame_num)


if __name__ == '__main__':
    if os.path.isfile("list_a_t.txt"):
        with open("list_a_t.txt", "rb") as fp:
            list_a_t = pickle.load(fp)  # len()=3449
    else:
        list_a_t = []

    #for i in range(213, 246):
    for i in range(0, total_backgrounds):
        if os.path.isfile('cnn_frame_num.txt'):
            cnn_frame_num = os.popen('cat cnn_frame_num.txt')
            cnn_frame_num = int(cnn_frame_num.read())
        else:
            cnn_frame_num = 0
        background_index = i
        root_dir = os.path.join(
            'matlab_frames_with_new_background/', 'background' + '_' + str(background_index) + '/')
        action_label(db, root_dir, cnn_frame_num, background_index)
