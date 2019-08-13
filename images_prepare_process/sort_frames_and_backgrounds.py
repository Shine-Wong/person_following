import cv2
import os
import random
import glob


def sort_background_pictures(start_index,background_path, sorted_background_path):
    '''this function is to sort all images in background_path according to their names and change their names in order to save in \
    sorted_background_path'''
    imgsname = glob.glob(background_path + '*')
    try:
         imgsname.sort()
    except ValueError:
         pass
    i = start_index
    for img_name in imgsname:
        img_new = cv2.imread(img_name).copy()
        rows,cols,depth=img_new.shape
        if rows==480 and cols==640:
            print("shape satisfied")
        else:
            img_new=cv2.resize(img_new, (640, 480))
        cv2.imwrite(sorted_background_path + str(i) + '.jpg', img_new)
        i += 1



if __name__ == '__main__':
    current_path=os.getcwd()
    background_path=current_path+'/background_pictures/'
    sorted_background_path=current_path+'/useful_background_with_index_sorted/'
    
    sort_background_pictures(start_index=0,background_path=background_path,sorted_background_path=sorted_background_path)
    print("has sort all images in background_pictures and save them in useful_background_with_index_sorted")

    frames_path = current_path + '/useful_frames/'
    sorted_frames_path = current_path + '/useful_frames_with_index_sorted/'
    sort_background_pictures(start_index=0,background_path=frames_path,sorted_background_path=sorted_frames_path)
    print("has sort all images in useful_frames and save them in useful_frames_with_index_sorted")
