# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import time

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    return img_files, mask_files, gt_files

def saveResult(Idx,img_file, img, boxes, dirname, verticals=None, texts=None):

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))


        sub_img = cv2.imread(img_file) #이미지 불러오는거 img_file에 해당 절대 경로가 들어가있을꺼임
        result2 = sub_img[58:137,126:260] #격자 기준으로 cut

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))


            cw = np.array(box).astype(np.int32)
            p_x = [cw[0,0],cw[1,0],cw[2,0],cw[3,0]]
            p_y = [cw[0,1],cw[1,1],cw[2,1],cw[3,1]]
            p_x.sort()
            p_y.sort()

            result = sub_img[p_y[0]:p_y[3], p_x[0]:p_x[3]]

            cv2.imwrite('{}/{}.jpg'.format(dirname,Idx) , result) #store Cropping image in recog_image/
            Idx = Idx + 1

        return Idx