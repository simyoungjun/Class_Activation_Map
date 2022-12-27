# %
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchsummaryX import summary





class CAM_Net(nn.Module):
    def __init__(self, batch_size):
        super(CAM_Net, self).__init__()
        self.fc = nn.Linear(512, 10)
        self.batch_size = batch_size
        self.vgg_sub = self.get_vgg_sub()
        
        
    def get_vgg_sub():
        vgg16 = models.vgg16(pretrained = True)
        return  nn.Sequential(*list(vgg16.children())[:-1])
        
    def forward(self, x):
        x = self.vgg_sub(x)
        x = x.view(self.batch_size, 512, 7*7).mean(2).view(self.batch_size,-1)
        # print(x.shape)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        print(x.shape)
        return x
    
#%%
if __name__=="__main__":

    batch_size = 32
    model = CAM_Net(32)
    summary(model,torch.zeros((batch_size,3,224,224)))


#%%

if __name__=="__main__":
    vgg16 = models.vgg16(pretrained=True) 
    mod = nn.Sequential(*list(vgg16.children())[:-1])   
    summary(mod, torch.zeros((3, 3, 224, 224)))
    
    batch_size = 32
    model=nn.Sequential(mod,CAM_Net(batch_size))
    summary(model,torch.zeros((batch_size,3,224,224)))
# %
