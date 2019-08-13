
import cv2
import os
import numpy as np
import json
import MySQLdb
import pickle
import conf


IP = conf.IP
PORT = conf.PORT
USER_NAME = conf.USER_NAME
PASSWD = conf.PASSWD
pretrain_DATABASE = conf.pretrain_DATABASE
TABLENAME = conf.TABLENAME

lower_shang = conf.lower_shang
upper_shang = conf.upper_shang

total_sequences_num = conf.sequences_num

root_dir = os.getcwd() + '/useful_frames_with_index_sorted/'
db = MySQLdb.connect(host=IP, port=PORT, user=USER_NAME,
                     passwd=PASSWD, db=pretrain_DATABASE)

def resize_and_get_mask(image):
    frm_hsv = cv2.cvtColor(cv2.resize(image, (80, 60)), cv2.COLOR_BGR2HSV)
    mask_shang = cv2.inRange(frm_hsv, lower_shang, upper_shang)
    #mask_xia = cv2.inRange(frm_hsv, lower_xia, upper_xia)
    #mask_full = mask_shang + mask_xia  # 1-channel image
    mask_full = mask_shang
    return mask_full
    
    
def resize_and_get_hsv(image):
    frm_hsv = cv2.cvtColor(cv2.resize(image, (80, 60)), cv2.COLOR_BGR2HSV)
    return frm_hsv
    

def action_label(db, root_dir):
    if os.path.isfile('cnn_frame_num.txt'):
        cnn_frame_num = os.popen('cat cnn_frame_num.txt')
        cnn_frame_num = int(cnn_frame_num.read())
    else:
        cnn_frame_num = 0
    # 0~275   483~565  700~795  806~849  854~953  1006~1139  1286~1305
    # 1313~1339   1389~1536  1838~1927  1963~2047  2055~2115  2197~ 2235
    # 2243~2329  2419, 2472  2480, 2585
    for i in range(0,total_sequences_num):
        frame1 = cv2.imread(os.path.join(root_dir, str(i) + '.jpg'))
        frame2 = cv2.imread(os.path.join(root_dir, str(i + 1) + '.jpg'))
        frame3 = cv2.imread(os.path.join(root_dir, str(i + 2) + '.jpg'))
        frame4 = cv2.imread(os.path.join(root_dir, str(i + 3) + '.jpg'))
        
        hsv_1=resize_and_get_hsv(frame1)
        hsv_2=resize_and_get_hsv(frame2)
        hsv_3=resize_and_get_hsv(frame3)
        hsv_4=resize_and_get_hsv(frame4)
        
        s_t = np.concatenate((hsv_1, hsv_2,
                        hsv_3, hsv_4), axis=2)

        mask_full_1 = resize_and_get_mask(frame1)
        mask_full_2 = resize_and_get_mask(frame2)
        mask_full_3 = resize_and_get_mask(frame3)
        mask_full_4 = resize_and_get_mask(frame4)

        print(mask_full_1.shape)
        #print(s_t)
        print(s_t.shape)

        nonzeroind1 = np.nonzero(mask_full_1)
        wide_mean1 = np.mean(nonzeroind1[1])
        nonzeroind2 = np.nonzero(mask_full_2)
        wide_mean2 = np.mean(nonzeroind2[1])
        nonzeroind3 = np.nonzero(mask_full_3)
        wide_mean3 = np.mean(nonzeroind3[1])
        nonzeroind4 = np.nonzero(mask_full_4)
        wide_mean4 = np.mean(nonzeroind4[1])

        frames_mean = (wide_mean1 + wide_mean2 + wide_mean3 + wide_mean4) / 4
        if frames_mean <= 35:
            a_t = np.array([1, 0, 0])
        if frames_mean > 35 and frames_mean < 45:
            a_t = np.array([0, 1, 0])
        if frames_mean >= 45:
            a_t = np.array([0, 0, 1])
        list_a_t.append(a_t)
        a_t_json = json.dumps(a_t.tolist())
        s_t_json = json.dumps(s_t.tolist())
        print(len(s_t_json))
        print(len(a_t_json))
        cnn_frame_num = cnn_frame_num + 1
        
        query = """INSERT INTO """+TABLENAME+""" (id,s_t,a_t) VALUES (%s, %s, %s)"""
        arg = (cnn_frame_num, s_t_json, a_t_json)
        cursor = db.cursor()
        cursor.execute(query, arg)
        db.commit()
        print("has committed to mysql")
    os.system('echo %d > cnn_frame_num.txt' % cnn_frame_num)
    with open("list_a_t.txt","wb") as fp:
        pickle.dump(list_a_t,fp)

        
if __name__ == '__main__':
    if os.path.isfile('list_a_t.txt'):
        with open("list_a_t.txt","rb") as fp:
            list_a_t=pickle.load(fp)
    else:
        list_a_t=[]
    action_label(db, root_dir)
