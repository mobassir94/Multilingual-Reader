#-*- coding: utf-8 -*-
"""
@authors: Mobassir Hossain,MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
import onnxruntime as ort

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class LangClassifier(object):
    def __init__(self,
                model_weights,
                providers=['CPUExecutionProvider'],
                img_dim=384,
                graph_input="input",
                labels=["ar","bn","en"],
                batch_size=32):
        self.img_dim=img_dim
        self.graph_input=graph_input
        self.model = ort.InferenceSession(model_weights, providers=providers)
        self.labels=labels
        self.batch_size=batch_size
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    T.Resize((img_dim, img_dim)),])
    
    def batch_infer(self,imgs):
        inps=[]
        for img in imgs:
            inp = self.transform(img)
            inp = inp.unsqueeze(0)
            inps.append(inp)
        inp_batch=torch.cat(inps)
        out=self.model.run(None,{self.graph_input:np.array(inp_batch)})
        preds=np.argmax(np.array(out).squeeze(), axis=1)
        return [self.labels[int(p)] for p in preds]

    def __call__(self,crops):
        langs=[]
        for idx in tqdm(range(0,len(crops),self.batch_size)):
            imgs=crops[idx:idx+self.batch_size]
            langs+=self.batch_infer(imgs)
        return langs