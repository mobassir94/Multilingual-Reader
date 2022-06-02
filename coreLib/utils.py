#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary,Mobassir Hossain
"""
from __future__ import print_function

#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import cv2 
import numpy as np
from PIL import Image, ImageEnhance
import gdown
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path

def download(id,save_dir):
    gdown.download(id=id,output=save_dir,quiet=False)

#------------------------------------
# region-utils 
#-------------------------------------
def intersection(boxA, boxB):
    # boxA=ref
    # boxB=sig
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    x_min,y_min,x_max,y_max=boxB
    selfArea  = abs((y_max-y_min)*(x_max-x_min))
    return interArea/selfArea
#---------------------------------------------------------------
def localize_box(box,region_boxes):
    '''
        lambda localization
    '''
    max_ival=0
    box_id=-1
    for idx,region_box in enumerate(region_boxes):
        ival=intersection(region_box,box)
        if ival==1:
            return idx
        if ival>max_ival:
            max_ival=ival
            box_id=idx
    if max_ival==0:
        return -1
    return box_id
#------------------------------------
# image-utils 
#-------------------------------------
#---------------------------------------------------------------
def remove_shadows(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm

def read_img(image):
    img=cv2.imread(image)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

#---------------------------------------------------------------
# recognition utils
#---------------------------------------------------------------
def padData(img,pad_loc,pad_dim,pad_type,pad_val):
    '''
        pads an image with white value
        args:
            img     :       the image to pad
            pad_loc :       (lr/tb) lr: left-right pad , tb=top_bottom pad
            pad_dim :       the dimension to pad upto
            pad_type:       central or left aligned pad
            pad_val :       the value to pad 
    '''
    
    if pad_loc=="lr":
        # shape
        h,w,d=img.shape
        if pad_type=="central":
            # pad widths
            left_pad_width =(pad_dim-w)//2
            # print(left_pad_width)
            right_pad_width=pad_dim-w-left_pad_width
            # pads
            left_pad =np.ones((h,left_pad_width,3))*pad_val
            right_pad=np.ones((h,right_pad_width,3))*pad_val
            # pad
            img =np.concatenate([left_pad,img,right_pad],axis=1)
        else:
            # pad widths
            pad_width =pad_dim-w
            # pads
            pad =np.ones((h,pad_width,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=1)
    else:
        # shape
        h,w,d=img.shape
        # pad heights
        if h>= pad_dim:
            return img 
        else:
            pad_height =pad_dim-h
            # pads
            pad =np.ones((pad_height,w,3))*pad_val
            # pad
            img =np.concatenate([img,pad],axis=0)
    return img.astype("uint8") 
#---------------------------------------------------------------
def padWords(img,dim,ptype="central",pvalue=255):
    '''
        corrects an image padding 
        args:
            img     :       numpy array of single channel image
            dim     :       tuple of desired img_height,img_width
            ptype   :       type of padding (central,left)
            pvalue  :       the value to pad
        returns:
            correctly padded image

    '''
    img_height,img_width=dim
    mask=0
    # check for pad
    h,w,d=img.shape
    w_new=int(img_height* w/h) 
    img=cv2.resize(img,(w_new,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    h,w,d=img.shape
    if w > img_width:
        # for larger width
        h_new= int(img_width* h/w) 
        img=cv2.resize(img,(img_width,h_new),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        # pad
        img=padData(img,
                     pad_loc="tb",
                     pad_dim=img_height,
                     pad_type=ptype,
                     pad_val=pvalue)
        mask=img_width

    elif w < img_width:
        # pad
        img=padData(img,
                    pad_loc="lr",
                    pad_dim=img_width,
                    pad_type=ptype,
                    pad_val=pvalue)
        if mask>img_width:
            mask=img_width
    
    # error avoid
    img=cv2.resize(img,(img_width,img_height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    return img,mask 
#---------------------------------------------------------------
# viz utils
#---------------------------------------------------------------
def draw_boxes_from_text_dict(image,text_dict):
    for crop_dict in text_dict:
        ln=crop_dict["line_no"]
        wn=crop_dict["word_no"]
        box=crop_dict["poly"]
        box = np.reshape(np.array(box), [-1,1,2]).astype(np.int64)
        image = cv2.polylines(image, [box], True,(255,0,0),2)
        x,y=box[0][0]
        image = cv2.putText(image,f"{ln}-{wn}",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return image

def draw_boxes(image,boxes):
    for idx,box in enumerate(boxes):
        box = np.reshape(np.array(box), [-1,1,2]).astype(np.int64)
        x,y=box[0][0]
        image = cv2.polylines(image, [box], True,(255,0,0),2)
        image = cv2.putText(image,str(idx),(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return image

    
