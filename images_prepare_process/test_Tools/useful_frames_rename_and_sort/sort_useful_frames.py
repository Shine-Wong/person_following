# -*- coding:utf-8 -*-

import os
from PIL import Image


class ImageRename():
    def __init__(self):
        self.path = '/home/wangshuai/test/useful_frames'
        self.output = 'useful_frames_with_index_sorted'

    def rename(self):
        filelist = os.listdir(self.path)
        filelist.sort()
        total_num = len(filelist)

        i = 0

        for item in filelist:
            if item.endswith('.jpg'):
                output_img_path = self.output + '/' + '%d' % i + '.jpg'
                #os.rename(src, dst)
                im = Image.open(self.path +  '/' + item)
                im.save(output_img_path)

                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i))


if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()