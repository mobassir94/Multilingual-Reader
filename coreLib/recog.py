#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function
# ---------------------------------------------------------
# imports
# ---------------------------------------------------------
import easyocr
# ---------------------------------------------------------
class BanOCR(object):
    def __init__(self):
        self.model=easyocr.Reader(['bn'],gpu=False,detector=False)

    def __call__(self,img,boxes):
        free_list=[]
        for box in boxes:
            x1,y1=box[0]
            x2,y2=box[1]
            x3,y3=box[2]
            x4,y4=box[3]
            free_list.append([[int(x1),int(y1)],
                            [int(x2),int(y2)],
                            [int(x3),int(y3)],
                            [int(x4),int(y4)]])

        texts=self.model.recognize(img,horizontal_list=[],free_list=free_list,detail=0)
        return texts
        