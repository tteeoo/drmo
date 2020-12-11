import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class ImageData(Dataset):
    """Class to represent image and label input."""

    def __init__(self, yx, width=48, height=48):
        self.width = width
        self.height = height
        y, x = yx 
        self.y = y # labels
        self.x = x # images

        # TODO: download data

    def __getitem__(self, index):
        """Process image and labels to train the model."""
    
        # Handle if the image is a path (training) or an OpenCV image (classifying)
        img = ''
        if type(self.x[index]) is str:
            img = Image.open(self.x[index])
        else:
            img = cv2.cvtColor(self.x[index], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        # Process the image
        img = img.resize((self.width, self.height)) 
        img = img.convert('RGB')
        img = np.asarray(img).transpose(-1, 0, 1)
        img = img/255
        img = torch.from_numpy(np.asarray(img)) # image tensor
        label = torch.from_numpy(np.asarray(self.y[index]).reshape([1, 1])) # label tensor
        
        return img, label, self.x[index]
    
    def __len__(self):
        return len(self.x)
