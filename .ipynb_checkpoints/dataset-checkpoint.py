import os

import numpy as np
import scipy.io
import cv2

from torch.utils.data import Dataset


class SBDDataset(Dataset):

    def __init__(self, root, split, transform=None):
        
        assert split in ['train', 'val', 'seg11valid'], "split should be one of ['train', 'val', 'seg11valid']"
        
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.data = open(os.path.join(self.root, f'{split}.txt'), 'r').read().splitlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_name = self.data[idx]
        img = cv2.imread(os.path.join(self.root, 'img', img_name + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mat = scipy.io.loadmat(os.path.join(self.root, 'cls', img_name + '.mat'))
        mask = mat['GTcls'][0, 0]['Segmentation']
        mask = mask[np.newaxis, ...]
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask