# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26},
#    number={7},
#    pages={3142-3155},
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately


import glob
import random
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

aug_times = 2
scales = [1, 0.9, 0.8, 0.7]

class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):

        batch_x = self.xs[index]
        '''
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        '''
        if self.sigma==0:
            sig=torch.randint(0,20, size=(batch_x.size(0),),dtype=batch_x.dtype)
            sig2=torch.randint(0,55, size=(batch_x.size(0),),dtype=batch_x.dtype)
        else:
            sig=self.sigma

        noise1 = torch.randn(batch_x.size(), dtype=batch_x.dtype).mul_(sig/255.0)
        noise2 = torch.randn(batch_x.size(), dtype=batch_x.dtype).mul_(sig2/255.0)
        batch_y = batch_x + noise1
        batch_y2 = batch_x + noise2

        return batch_y, batch_x, batch_y2

    def __len__(self):
        return self.xs.size(0)

class ValDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, data_dir, sigma):
        super(ValDataset, self).__init__()
        self.files = glob.glob(data_dir+'/*.png')
        self.sigma = sigma

    def __getitem__(self, index):
        file_path = self.files[index]
        batch_x = cv2.imread(file_path, 0)
        noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = batch_x + noise

        return batch_y, batch_x

    def __len__(self):
        return len(self.files)

def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name, patch_size, stride):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/Train400', batch_size=128, patch_size=80, stride=10, verbose=True):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    #random.shuffle(file_list)
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i], patch_size, stride)
        for patch in patches:
            if patch.shape[0] is patch_size and patch.shape[1] is patch_size:
                data.append(patch)
            #data.append(patch)
        if verbose:
            print(file_list[i] + ' is done')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('--finished loading training data--')
    return data

def im_to_gray(data_dir='data/Train400', save_dir=''):
    file_list = glob.glob(data_dir+'/*.png')
    for i in range(len(file_list)):
        print(file_list[i])
        img = cv2.imread(file_list[i], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(file_list[i].replace(data_dir, save_dir), img)

if __name__ == '__main__':

    #data = datagenerator(data_dir='data/Train400')
    im_to_gray('F:\DIV2K\DIV2K_train_HR', 'F:\DIV2K\DIV2k_train_HR_gray')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')
