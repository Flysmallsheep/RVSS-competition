import re
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from os import path

class SteerDataSet(Dataset):
    """
    Dataset that can be constructed from:
      - root_folder (recursively gathers images), OR
      - an explicit list of filenames (recommended for train/val split)
    It crops the top 120 px, applies transform, and returns (image_tensor, class_id).
    """

    def __init__(self, root_folder=None, img_ext=".jpg", transform=None, filenames=None, recursive=True):
        self.transform = transform
        self.img_ext = img_ext
        self.recursive = recursive
        self.totensor = transforms.ToTensor()

        self.class_labels = [
            "sharp left",
            "left",
            "straight",
            "right",
            "sharp right",
        ]

        if filenames is not None:
            self.filenames = list(filenames)
        else:
            if root_folder is None:
                raise ValueError("Provide either root_folder or filenames.")
            root = Path(root_folder)
            if recursive:
                self.filenames = [str(p) for p in root.rglob(f"*{img_ext}")]
            else:
                self.filenames = [str(p) for p in root.glob(f"*{img_ext}")]

        self.filenames = sorted(self.filenames)

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
