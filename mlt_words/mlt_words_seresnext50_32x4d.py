# Import basic packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import zipfile
from torch import onnx
from sklearn import metrics
# Import PyTorch packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import os, random, shutil
import timm 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss
from tqdm.notebook import tqdm
import random
import seaborn as sns
sns.set_style("darkgrid")
from torchvision import models 
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from albumentations import Compose, Normalize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
import warnings
warnings.filterwarnings("ignore")

bn = len(os.listdir("/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/bn_words"))
en = len(os.listdir("/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/en_words"))
ar = len(os.listdir("/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/ar_words"))
print(bn,en,ar)

train =  1# 1 = train, 0 = inference
image_size = 384
batch_size= 80
no_epochs = 10
sample_size = 250000 
split_percent = 0.20
max_patience = 3
num_workers = 1
lr = 0.00003
lr1 = 0.000003
model_name = 'seresnext50_32x4d'
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

train_dir = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words'

if os.path.exists(train_dir+'/train'):
    shutil.rmtree(train_dir+'/train')
if os.path.exists(train_dir+'/val'):
    shutil.rmtree(train_dir+'/val')

    
os.makedirs(train_dir+'/train', exist_ok=True)
os.makedirs(train_dir+'/val', exist_ok=True)

os.makedirs(train_dir+'/train/ar_words', exist_ok=True)
os.makedirs(train_dir+'/val/ar_words', exist_ok=True)

os.makedirs(train_dir+'/train/bn_words', exist_ok=True)
os.makedirs(train_dir+'/val/bn_words', exist_ok=True)

os.makedirs(train_dir+'/train/en_words', exist_ok=True)
os.makedirs(train_dir+'/val/en_words', exist_ok=True)

def custom_random_sampler(source = '/content/train/mlt_words/' ,destination_train = '/content/sample_data',destination_val = '/content/sample_data',sample_size = 100):
    all_data = []
    if(len(os.listdir(source)) < sample_size):
        sample_size = len(os.listdir(source))
        print("updated sample size = ",sample_size)
    source_data = os.listdir(source)
    random.shuffle(source_data)
    print("our sample size = ",sample_size)
    for idx,data in enumerate(source_data):
        if(idx>sample_size):
            break
        all_data.append(source+data)
    
    for i in range(len(all_data)):
        shutil.copy(f'{all_data[i]}', destination_train)
        
    random.shuffle(all_data)
    val = int(len(all_data)*split_percent)
    all_train = os.listdir(destination_train)
    for i in range(val):
        shutil.move(destination_train+'/'+all_train[i], destination_val)
    

custom_random_sampler(source = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/ar_words/' ,
                      destination_train = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/train/ar_words',
                     destination_val = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/val/ar_words',
                     sample_size = sample_size)
custom_random_sampler(source = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/bn_words/' ,
                      destination_train = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/train/bn_words',
                     destination_val = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/val/bn_words',
                     sample_size = sample_size)
custom_random_sampler(source = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/en_words/' ,
                      destination_train = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/train/en_words',
                     destination_val = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/val/en_words',
                     sample_size = sample_size)
train_dir = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/train'
test_dir = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/val'
save_dir = '/home/apsisdev/mobassir/data/mlt_reader_research/multilingual_words/'
# Make sure all images have size [image_size, image_size]

def get_train_transforms():
    return Compose([
            Resize(image_size, image_size),
            Transpose(p=0.5),
            #HorizontalFlip(p=0.5),
            #VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.2),
            Cutout(p=0.1),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            Resize(image_size, image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

# train_transform = get_train_transforms()
# transform = get_valid_transforms()

train_transform = T.Compose([
    
    T.Resize((image_size,image_size)), 
    #T.RandomPerspective(),
    #T.RandomHorizontalFlip(),
    #T.RandomVerticalFlip(),
    
    T.AutoAugment(),
    
    #T.FiveCrop(image_size),
    #T.TenCrop(image_size),
    
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  
    
])

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Resize((image_size, image_size)),
    
])


# Load train data and test data
train_data = ImageFolder(root=train_dir, transform=train_transform)
validate_data = ImageFolder(root=test_dir, transform=transform)

# Store data in ImageFolder to DataLoader
train_ds = DataLoader(train_data,batch_size,  shuffle=True, pin_memory=False, num_workers=num_workers,persistent_workers = True)#,persistent_workers = True
validate_ds = DataLoader(validate_data,batch_size, shuffle=True, pin_memory=False, num_workers=num_workers,persistent_workers = True)

# View train dataset size
print("Train dataset has {0} images".format(len(train_data)))

# View valid dataset size
print("Validation dataset has {0} images".format(len(validate_data)))

# View image size and class
fst_img, fst_lbl = train_data[0]
print("First image has size: {0} and class: {1}.".format(fst_img.shape, fst_lbl))

sc_img, sc_lbl = train_data[10]
print("Another random image has size: {0} and class: {1}.".format(sc_img.shape, sc_lbl))

# View all classes
classes = train_data.classes
print("There are {0} classes in total: ".format(len(classes)))
print(classes)
'''
for images, _ in validate_ds:
    print('images.shape:', images.shape)
    plt.figure(figsize=(30,30))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))
    break
'''
def get_model(model_name=None,pretrained = True):
    #model = models.efficientnet_b4(pretrained = pretrained)#resnext50_32x4d
    
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=len(classes))#, num_classes=len(classes)
#     for q in model.parameters():
#         q.requires_grad = False
    #print(model)
#     model.head.fc = nn.Sequential(
#       nn.Linear(in_features=model.head.fc.in_features, out_features=model.head.fc.in_features//2) ,
#       nn.ReLU(),
#       nn.Linear(in_features=(model.head.fc.in_features)//2, out_features=model.head.fc.in_features//4) ,
#       nn.ReLU(),
#       nn.Dropout(p=0.1), 
#       nn.Linear(in_features=model.head.fc.in_features//4 , out_features=len(classes)),
#       nn.LogSoftmax(dim=1)
#     )
    return model

model = get_model(model_name = model_name,pretrained = True)
#model

# https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes = len(classes), smoothing=0.0, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
# Perform training and cross validation
# Calculate accuracy
def calcAccuracy(scores, label):
    _, prediction = torch.max(scores, dim=1)
    return torch.tensor(torch.sum(prediction == label).item()/len(scores))

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5):
        """
        :param patience: how many epochs to wait before stopping when acc is
               not improving

        """
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
    def __call__(self, val_acc):
        if self.best_acc == None:
            self.best_acc = val_acc
        elif val_acc > self.best_acc:
            self.best_acc = val_acc
            # reset counter if validation acc improves
            self.counter = 0
        else:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
#@torch.no_grad()
# Cross validate
def validate(validate_ds, model,softmax):
    model.eval()
    validate_length = 0
    accuracy = 0
    label = []
    preds = []
    for img, lbl in validate_ds:
        with torch.no_grad():
            scores = model(img)
            loss = softmax(scores, lbl)
            lbl_gpu = lbl
            lbl = lbl.detach().cpu()
            _, prediction = torch.max(scores, dim=1)
            prediction = prediction.detach().cpu()
            label.append(lbl)
            preds.append(prediction)
            accuracy += calcAccuracy(scores, lbl_gpu)
            validate_length += 1
            
    accuracy /= validate_length
    print("accuracy -> ",accuracy)
    y_true = torch.cat(label, dim=0).numpy()
    y_pred = torch.cat(preds, dim=0).numpy()
    print('recall_score = ',metrics.recall_score(y_true, y_pred, average='macro'))
    print('precision_score = ',metrics.precision_score(y_true, y_pred, average='macro'))
 
    macro = metrics.f1_score(y_true, y_pred, average='macro')
    #print(macro)
    return loss, macro

# Run the training and cross validation
def fit(train_ds, validate_ds, no_epochs, optimizer, model):
    model.train()
    history = []
    softmax = LabelSmoothingLoss(smoothing=0.01) #nn.CrossEntropyLoss(reduction  = 'mean')
    valid_acc = 0
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience = max_patience)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=lr1,cycle_momentum=False)
    early_stopping = EarlyStopping(patience = max_patience)
    for index in range(no_epochs):
        print("starting epoch-> ",index)
        # Train
#         if(index<3):
#             for q in model.parameters():
#                 q.requires_grad = False
#         else:
#             for q in model.parameters():
#                 q.requires_grad = True
        model = model.to(device)
        #tk = tqdm(train_ds, total=int(len(train_ds.ds)))
        #for idx, (img,lbl) in enumerate(tk):
        for img, lbl in train_ds:
            with torch.cuda.amp.autocast(enabled=use_amp):
                scores = model(img)
                loss = softmax(scores, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # set_to_none=True here can modestly improve performance
            scheduler.step()
            #print(loss.item())
            #tk.set_postfix(loss=loss.item())
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
            
        # Validate
        valid_loss, macro = validate(validate_ds, model, softmax)
        #scheduler.step(macro)
            
        # Print epoch record
        print(f"Epoch [{index + 1}/{no_epochs}] => loss: {loss}, val_loss: {valid_loss}, Validation F1 Macro: {macro}")
        if(macro > valid_acc):
          print("----------->>> val F1 improved, saving best weight....")
          torch.save(model.state_dict(), f'{save_dir}/model.pth')
          valid_acc = macro
        history.append({"loss": loss,
                       "valid_loss": valid_loss,
                       "F1 Macro": macro
                       })
       
        early_stopping(macro)
        if early_stopping.early_stop:
            break
    return history

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class to_GPU():
    def __init__(self, ds, device):
        self.ds = ds
        self.device = device
    
    def __iter__(self):
        for batch in self.ds:
            yield to_device(batch, self.device)
            
# Initialize model and data before training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_ds = to_GPU(train_ds, device)
validate_ds = to_GPU(validate_ds, device)
print(device)


if __name__ == "__main__":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if(train):
        history = fit(train_ds, validate_ds, no_epochs, optimizer, model)
        print(history)
    
'''
(mobassir) apsisdev@ML:~/mobassir/data/mlt_reader_research/multilingual_words$ python mlt_words_seresnext50_32x4d.py 
1150586 438948 239767
updated sample size =  239767
our sample size =  239767
our sample size =  250000
our sample size =  250000
Train dataset has 591816 images
Validation dataset has 147953 images
First image has size: torch.Size([3, 384, 384]) and class: 0.
Another random image has size: torch.Size([3, 384, 384]) and class: 0.
There are 3 classes in total: 
['ar_words', 'bn_words', 'en_words']
cuda
starting epoch->  0
accuracy ->  tensor(0.9970)
recall_score =  0.9969607273788919
precision_score =  0.9969382506583792
Epoch [1/10] => loss: 0.06398974359035492, val_loss: 0.06689220666885376, Validation F1 Macro: 0.9969491172634622
----------->>> val F1 improved, saving best weight....
starting epoch->  1
accuracy ->  tensor(0.9975)
recall_score =  0.9975022715297618
precision_score =  0.9975015375392245
Epoch [2/10] => loss: 0.07044508308172226, val_loss: 0.07338831573724747, Validation F1 Macro: 0.9975018980539875
----------->>> val F1 improved, saving best weight....
starting epoch->  2
accuracy ->  tensor(0.9980)
recall_score =  0.9980058498946885
precision_score =  0.998004205357024
Epoch [3/10] => loss: 0.0654323473572731, val_loss: 0.0630190297961235, Validation F1 Macro: 0.9980047876530732
----------->>> val F1 improved, saving best weight....
starting epoch->  3
accuracy ->  tensor(0.9983)
recall_score =  0.9982580870157585
precision_score =  0.9982427236543026
Epoch [4/10] => loss: 0.0638454258441925, val_loss: 0.06309101730585098, Validation F1 Macro: 0.9982502248685586
----------->>> val F1 improved, saving best weight....
starting epoch->  4
accuracy ->  tensor(0.9978)
recall_score =  0.997760363689446
precision_score =  0.997725735330515
Epoch [5/10] => loss: 0.06669430434703827, val_loss: 0.06349916011095047, Validation F1 Macro: 0.9977420711476501
INFO: Early stopping counter 1 of 3
starting epoch->  5
accuracy ->  tensor(0.9980)
recall_score =  0.9980370699087301
precision_score =  0.998027481794856
Epoch [6/10] => loss: 0.06403135508298874, val_loss: 0.06301116943359375, Validation F1 Macro: 0.9980322103938634
INFO: Early stopping counter 2 of 3
starting epoch->  6
accuracy ->  tensor(0.9981)
recall_score =  0.9980587352894154
precision_score =  0.9980715907237186
Epoch [7/10] => loss: 0.06312955170869827, val_loss: 0.06300082802772522, Validation F1 Macro: 0.9980647976639202
INFO: Early stopping counter 3 of 3
INFO: Early stopping
[{'loss': tensor(0.0640, device='cuda:0', grad_fn=<MeanBackward0>), 'valid_loss': tensor(0.0669, device='cuda:0'), 'F1 Macro': 0.9969491172634622}, {'loss': tensor(0.0704, device='cuda:0', grad_fn=<MeanBackward0>), 'valid_loss': tensor(0.0734, device='cuda:0'), 'F1 Macro': 0.9975018980539875}, {'loss': tensor(0.0654, device='cuda:0', grad_fn=<MeanBackward0>), 'valid_loss': tensor(0.0630, device='cuda:0'), 'F1 Macro': 0.9980047876530732}, {'loss': tensor(0.0638, device='cuda:0', grad_fn=<MeanBackward0>), 'valid_loss': tensor(0.0631, device='cuda:0'), 'F1 Macro': 0.9982502248685586}, {'loss': tensor(0.0667, device='cuda:0', grad_fn=<MeanBackward0>), 'valid_loss': tensor(0.0635, device='cuda:0'), 'F1 Macro': 0.9977420711476501}, {'loss': tensor(0.0640, device='cuda:0', grad_fn=<MeanBackward0>), 'valid_loss': tensor(0.0630, device='cuda:0'), 'F1 Macro': 0.9980322103938634}, {'loss': tensor(0.0631, device='cuda:0', grad_fn=<MeanBackward0>), 'valid_loss': tensor(0.0630, device='cuda:0'), 'F1 Macro': 0.9980647976639202}]
(mobassir) apsisdev@ML:~/mobassir/data/mlt_reader_research/multilingual_words$ 

'''




  
