import os
import numpy as np
import torch
import torch.utils.data as Data
import h5py
import json
from split import DatasetSplit
import skimage.io as sio

TEST_SIZE = 0.3

class TrainDataset(Data.Dataset):
    def __init__(self, pix_dir, test=False):
        super(TrainDataset, self).__init__()
        self.CWD_PATH = pix_dir
        with open(pix_dir+'/pix3d.json', 'r') as f:
            self.dataSets = json.load(f)

    def __getitem__(self, index):
        meshModel =
        image = sio.imread(self.dataSets[index]['img'])
        return image, voxel

    def __len__(self):
        return len(self.dataSets)