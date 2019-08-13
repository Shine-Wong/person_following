import cv2
import numpy as np
import matplotlib.pyplot as plt
root_dir = '/home/wangshuai/pycharmPro/following_robot/images_prepare_process/test_Tools/useful_frames_sort_test/useful_frames_with_index_sorted/21.jpg'
frame = cv2.imread(root_dir)
lower_shang = np.array([120,30,30])
upper_shang = np.array([190,255,255])

frame = cv2.resize(frame, (80, 60))
frm_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

mask_shang = cv2.inRange(frm_hsv, lower_shang, upper_shang)
mask_full = mask_shang

nonzeroind1 = np.nonzero(mask_full)
wide_mean1 = np.mean(nonzeroind1[1])
#plt.imshow(nonzeroind1,cmap='gray')

img = cv2.bitwise_and(frame,frame,mask=mask_full)
plt.imshow(mask_full,cmap='gray')
#print(frm_hsv)
#plt.imshow(img)
#plt.imshow(img[...,::-1],interpolation='bicubic')
#print(wide_mean1)
#plt.imshow(frm_hsv)
#plt.imshow(mask_full)

plt.show()