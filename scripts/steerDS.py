import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path
"""
Summary
1. Finds all images in a folder
2. Loads each image and crops the top 120 pixels
3. Applies transforms (resize, normalize, etc.)
4. Extracts the steering angle from the filename
5. Converts the continuous angle to a discrete class (0-4)
6. Returns (image_tensor, class_label) for training
7. This enables using PyTorch's DataLoader for batching, shuffling, and parallel loading during training.
"""

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()
        self.class_labels = ['sharp left',
                            'left',
                            'straight',
                            'right',
                            'sharp right']
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)[120:, :, :] # crop the image to the bottom half to remove the sky, focus on the track
        
        if self.transform == None:
            img = self.totensor(img) # convert the image to a tensor, vector of pixels
        else:
            img = self.transform(img)   
        
        steering = path.split(f)[-1].split(self.img_ext)[0][6:]
        steering = float(steering)       

        if steering <= -0.5:
            steering_cls = 0
        elif steering < 0:
            steering_cls = 1
        elif steering == 0:
            steering_cls = 2
        elif steering < 0.5:
            steering_cls = 3
        else:
            steering_cls = 4 
                      
        return img, steering_cls
