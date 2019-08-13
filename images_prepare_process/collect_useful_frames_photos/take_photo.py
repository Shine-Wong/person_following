#! -*- coding:utf-8 -*-
import cv2
import time
import numpy as np
import os
import datetime

cap = cv2.VideoCapture(0)
label = "m"
# 读取摄像头
while(1):
    # get frame 读取图像
    ret, frame= cap.read()
    # show a frame
    cv2.imshow("capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
    #if cv2.waitKey(1) & 0xFF == k:
        time_now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        print(type(time_now))
        if time_now[20] != label:
            label = time_now[20]
            print(time_now)
            cv2.imwrite('/home/nuc01/photos/saved_frames/' + np.str(time_now) + '.jpg', frame)

        #break
cap.release()
cv2.destroyAllWindoqws()