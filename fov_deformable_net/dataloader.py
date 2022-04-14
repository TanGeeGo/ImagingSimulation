import glob
import numpy as np
import torch
import random
import h5py
import time
import torch.utils.data as data

class Dataset_from_h5(data.Dataset):

    def __init__(self, src_path, recrop_patch_size=128, sigma=5, train=True):
        self.path = src_path
        self.recrop_size = recrop_patch_size
        self.sigma = sigma
        self.if_train = train
        h5f = h5py.File(self.path, 'r')
        self.keys = list(h5f.keys())
        if self.if_train:
            random.shuffle(self.keys)
        h5f.close()

    def __getitem__(self, index):
        h5f = h5py.File(self.path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key]).reshape(h5f[key].shape)
        h5f.close()

        if self.if_train:
            (H, W, C) = data.shape
            rnd_h = random.randint(0, max(0, H - self.recrop_size))
            rnd_w = random.randint(0, max(0, W - self.recrop_size))
            patch = data[rnd_h:rnd_h + self.recrop_size, rnd_w:rnd_w + self.recrop_size, :]

            p = 0.5
            if random.random() > p: #RandomRot90
                patch = patch.transpose(1, 0, 2)
            if random.random() > p: #RandomHorizontalFlip
                patch = patch[:, ::-1, :]
            if random.random() > p: #RandomVerticalFlip
                patch = patch[::-1, :, :]
        else:
            patch = data

        input = patch[:, :, 0:3]
        if self.sigma:
            noise = np.random.normal(loc=0, scale=self.sigma/255.0, size=input.shape)
            input = input + noise
            input = np.clip(input, 0.0, 1.0)

        label = patch[:, :, 3:6]
        fld_info = patch[:, :, 6:8]

        input_wz_fld = np.concatenate([input, fld_info], 2)

        input = torch.from_numpy(np.ascontiguousarray(np.transpose(input_wz_fld, (2, 0, 1)))).float()
        label = torch.from_numpy(np.ascontiguousarray(np.transpose(label, (2, 0, 1)))).float()

        return input, label

    def __len__(self):
        return len(self.keys)
