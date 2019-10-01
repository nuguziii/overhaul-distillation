import numpy as np
import torch
import os
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import models.DnCNN as dncnn
import distiller
import cv2

def strToBool(str):
	return str.lower() in ('true', 'yes', 'on', 't', '1')

#DnCNN(batch_size=128, n_epoch=150, sigma=0, lr=1e-3, lr2=2e-5, depth=20, device="cuda:0", data_dir='./data/Train400', model_dir='models', number=0)
import argparse
parser = argparse.ArgumentParser()
parser.register('type', 'bool', strToBool)
parser.add_argument('--sigma', type=int, default=25)
parser.add_argument('--testset', type=int, default=25, help='1: Urban100 / 2: Set14 / 3: Set5 / 4: Manga109 / 5: B100')
parser.add_argument('--data_dir', default='F:\Test')
parser.add_argument('--model_dir', default='F:\models\model_0905\\net.pth')
parser.add_argument('--save_dir', default='F:\results')
param = parser.parse_args()

def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def save_residual(r, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]

    r = 2 * (r + 0.4) - 0.3
    imsave(path, np.clip(r, 0, 1))


def save_structure(s, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]

    # s = 4*(s+0.3)-0.7
    s = 1.8 * (s + 0.7) - 0.8
    imsave(path, np.clip(s, 0, 1))


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

set_dir = param.data_dir
set_list = {1: "Urban100", 2: "Set14", 3: "Set5", 4: "Manga109", 5: "B100"}
set = set_list[param.testset]
sigma = param.sigma
result_dir = param.save_dir
model_dir = param.model_dir

model = torch.load(model_dir)
'''
model = dncnn.DnCNN()
print(model)
weight_dict_ = checkpoint['state_dict']
weight_dict = {}
for k,v in weight_dict_.items():
    weight_dict[k[7:]] = v
model.load_state_dict(weight_dict)
'''
model.eval()

if torch.cuda.is_available():
    model = model.cuda()

if not os.path.exists(os.path.join(result_dir, set)):
    os.mkdir(os.path.join(result_dir, set))
psnrs = []
ssims = []

for im in os.listdir(os.path.join(set_dir, set)):
    if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
        x = np.array(cv2.imread(os.path.join(set_dir, set, im), cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255.0
        init_shape = x.shape

        np.random.seed(seed=0)
        y = x+np.random.normal(0, sigma/255.0, x.shape)
        y = y.astype(np.float32)
        y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

        torch.cuda.synchronize()
        y_ = y_.cuda()
        x_ = model(y_)

        x_ = x_.view(y.shape[0], y.shape[1])
        x_ = x_.cpu()
        x_ = x_.detach().numpy().astype(np.float32)

        torch.cuda.synchronize()
        print('%10s : %10s' % (set, im))
        psnr_x_ = compare_psnr(x, x_)
        ssim_x_ = compare_ssim(x, x_)

        psnrs.append(psnr_x_)
        ssims.append(ssim_x_)

        name, ext = os.path.splitext(im)
        #show(np.hstack((y, x_)))  # show the image
        save_result(x_, path=os.path.join(result_dir, set, name + '_d' + ext))

psnr_avg = np.mean(psnrs)
ssim_avg = np.mean(ssims)
psnrs.append(psnr_avg)
ssims.append(ssim_avg)

print('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set, psnr_avg, ssim_avg))

