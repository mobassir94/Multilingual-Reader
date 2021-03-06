#-*- coding: utf-8 -*-
"""
@author:Mobassir Hossain,MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
from .utils import localize_box,LOG_INFO,download
from .rotation import auto_correct_image_orientation
from .detector import Detector
from .classifier import LangClassifier
from .recog import BanOCR

from paddleocr import PaddleOCR
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import copy
import pandas as pd

#-------------------------
# class
#------------------------

class MultilingualReader(object):
    def __init__(self,
                 lang_onnx="weights/lang.onnx",
                 lang_gid="1eJGL6nP_dXapuaDCeObHXcxb9IPejas5"):
        self.line_en=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet')
        self.word_ar=PaddleOCR(use_angle_cls=True, lang='ar')
        self.det=Detector()
        LOG_INFO("Loaded Detector")
        if not os.path.exists(lang_onnx):
            download(lang_gid,lang_onnx)
        self.lang=LangClassifier(lang_onnx)
        LOG_INFO("Loaded Language classifier")
        self.bocr=BanOCR()
        LOG_INFO("Loaded EasyOCR BN(Temporary)")
        

        


    #-------------------------------------------------------------------------------------------------------------------------
    # exectutives
    #-------------------------------------------------------------------------------------------------------------------------
    def execite_rotation_fix(self,image):
        result= self.line_en(image,rec=False)
        image,mask,angle=auto_correct_image_orientation(image,result)
        # -- coverage
        h,w,_=image.shape
        idx=np.where(mask>0)
        y1,y2,x1,x2 = np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1])
        ht=y2-y1
        wt=x2-x1
        coverage=round(((ht*wt)/(h*w))*100,2)  

        rot_info={"operation":"rotation-fix",
                  "optimized-angle":angle,
                  "text-area-coverage":coverage}

        return image,rot_info

    def process_boxes(self,word_boxes,line_boxes):
        # line_boxes
        line_orgs=[]
        line_refs=[]
        for bno in range(len(line_boxes)):
            tmp_box = copy.deepcopy(line_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            line_orgs.append([x1,y1,x2,y2])
            line_refs.append([x1,y1,x2,y2])
        
        # merge
        for lidx,box in enumerate(line_refs):
            if box is not None:
                for nidx in range(lidx+1,len(line_refs)):
                    x1,y1,x2,y2=box    
                    x1n,y1n,x2n,y2n=line_orgs[nidx]
                    dist=min([abs(y2-y1),abs(y2n-y1n)])
                    if abs(y1-y1n)<dist and abs(y2-y2n)<dist:
                        x1,x2,y1,y2=min([x1,x1n]),max([x2,x2n]),min([y1,y1n]),max([y2,y2n])
                        box=[x1,y1,x2,y2]
                        line_refs[lidx]=None
                        line_refs[nidx]=box
                        
        line_refs=[lr for lr in line_refs if lr is not None]
                                
        # word_boxes
        word_refs=[]
        for bno in range(len(word_boxes)):
            tmp_box = copy.deepcopy(word_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            word_refs.append([x1,y1,x2,y2])
            
        
        data=pd.DataFrame({"words":word_refs,"word_ids":[i for i in range(len(word_refs))]})
        # detect line-word
        data["lines"]=data.words.apply(lambda x:localize_box(x,line_refs))
        data["lines"]=data.lines.apply(lambda x:int(x))
        # register as crop
        text_dict=[]
        for line in data.lines.unique():
            ldf=data.loc[data.lines==line]
            _boxes=ldf.words.tolist()
            _bids=ldf.word_ids.tolist()
            _,bids=zip(*sorted(zip(_boxes,_bids),key=lambda x: x[0][0]))
            for idx,bid in enumerate(bids):
                _dict={"line_no":line,"word_no":idx,"crop_id":bid,"poly":word_boxes[bid]}
                text_dict.append(_dict)
        data=pd.DataFrame(text_dict)
        return data,text_dict


    def __call__(self,img_path,exec_rot=False):
        executed=[]
        # -----------------------start-----------------------
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # orientation
        if exec_rot:
            img,rot_info=self.execite_rotation_fix(img)
            executed.append(rot_info)
        # text detection
        line_boxes,_=self.det.detect(img,self.line_en)
        word_boxes,crops=self.det.detect(img,self.word_ar)
        # line-word sorting
        df,text_dict=self.process_boxes(word_boxes,line_boxes)
        # language classification
        cids=df.crop_id.tolist()
        word_crops=[crops[i] for i  in cids]
        langs=self.lang(word_crops)
        df["lang"]=langs 
        # language division
        adf=df.loc[df.lang=="ar"]
        bdf=df.loc[df.lang=="bn"]
        edf=df.loc[df.lang=="en"]

        #--------------------------------english------------------------------------
        en_ids=edf.crop_id.tolist()
        if len(en_ids)>0:
            en_crops=[crops[idx] for idx in en_ids]
            res_eng = self.line_en.ocr(en_crops,det=False,cls=True)
            en_text=[i for i,_ in res_eng]
            edf["text"]=en_text
        #--------------------------------arabic------------------------------------
        ar_ids=adf.crop_id.tolist()
        if len(ar_ids)>0:
            ar_crops=[crops[idx] for idx in ar_ids]
            res_arb = self.word_ar.ocr(ar_crops,det=False,cls=True)
            ar_text=[i for i,_ in res_arb]
            adf["text"]=ar_text
        #--------------------------------bangla------------------------------------
        bn_ids=bdf.crop_id.tolist()
        if len(bn_ids)>0:
            bn_boxes=[word_boxes[idx] for idx in bn_ids]
            bn_text=self.bocr(img,bn_boxes)
            bdf["text"]=bn_text

        df=pd.concat([edf,adf,bdf],ignore_index=True)
        return img,df,text_dict

        
        