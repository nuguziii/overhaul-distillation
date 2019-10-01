import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import _Loss
import torch.optim as optim
from skimage.measure import compare_psnr
import numpy as np
import gc, cv2

import models.MobileNet as Mov
import models.ResNet as ResNet
import models.DnCNN as dncnn
import distiller

import data_generator as dg
from data_generator import DenoisingDataset, ValDataset

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch ImageNet-1k Training')
parser.add_argument('--data_path', type=str, help='path to dataset')
parser.add_argument('--net_type', default='dncnn', type=str, help='networktype: resnet, mobilenet, dncnn')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=500, type=int, help='print frequency (default: 500)')
parser.add_argument('--sigma', default=0, type=int, help='noise level(0,25,50)')

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum')/2

best_acc = 0

def main():
    global args, best_acc
    args = parser.parse_args()

    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'val')

    if args.net_type == 'mobilenet':
        t_net = ResNet.resnet50(pretrained=True)
        s_net = Mov.MobileNet()
    elif args.net_type == 'resnet':
        t_net = ResNet.resnet152(pretrained=True)
        s_net = ResNet.resnet50(pretrained=False)
    elif args.net_type == 'dncnn':
        t_net = dncnn.DnCNN()
        s_net = dncnn.DnCNN()
    else:
        print('undefined network type !!!')
        raise

    d_net = distiller.Distiller(t_net, s_net)

    print ('Teacher Net: ')
    print(t_net)
    print ('Student Net: ')
    print(s_net)
    print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
    print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

    t_net = torch.nn.DataParallel(t_net).cuda()
    s_net = torch.nn.DataParallel(s_net).cuda()
    d_net = torch.nn.DataParallel(d_net).cuda()

    # define loss function (criterion) and optimizer
    #criterion_CE = nn.CrossEntropyLoss().cuda()
    criterion_CE = sum_squared_error()
    optimizer = optim.Adam(s_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, 0.999))
    #optimizer = torch.optim.SGD(list(s_net.parameters()) + list(d_net.module.Connectors.parameters()), args.lr,
    #                            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    cudnn.benchmark = True

    for epoch in range(1, args.epochs+1):
        x = dg.datagenerator(data_dir=traindir, batch_size=args.batch_size, patch_size=48, stride=30, verbose=False) / 255.0
        x = torch.from_numpy(x.transpose((0, 3, 1, 2))).type(torch.FloatTensor)
        train_loader = torch.utils.data.DataLoader(dataset=DenoisingDataset(x, args.sigma), num_workers=args.workers,
                                                   drop_last=True, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True, sampler=None)

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_with_distill(train_loader, d_net, optimizer, criterion_CE, epoch)
        # evaluate on validation set
        acc1 = validate(valdir, s_net, epoch, 50)

        # remember best prec@1 and save checkpoint
        is_best = acc1 >= best_acc
        best_acc = max(acc1, best_acc)
        print ('Current validation set best accuracy:', best_acc)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': s_net.state_dict(),
            'best_acc': acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, s_net)
        gc.collect()

    print ('Best accuracy:', best_acc)


def validate(valdir, model, epoch, sigma):
    model.eval()

    psnrs=[]
    for im in os.listdir(valdir):
        if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = np.array(cv2.imread(os.path.join(valdir, im), cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255.0

            np.random.seed(seed=0)
            y = x + np.random.normal(0, sigma / 255.0, x.shape)
            y = y.astype(np.float32)
            y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

            torch.cuda.synchronize()
            y_ = y_.cuda()
            x_ = model(y_)

            x_ = x_.view(y.shape[0], y.shape[1])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)

            torch.cuda.synchronize()
            psnr_x_ = compare_psnr(x, x_)

            psnrs.append(psnr_x_)

    psnr_avg = np.mean(psnrs)

    print('* Epoch: [{0}/{1}]\t PSNR {acc:.3f}dB'.format(epoch, args.epochs, acc=psnr_avg))
    return psnr_avg

def test(testdir, model, epoch, sigma):
    model.eval()

    psnrs=[]
    for im in os.listdir(testdir):
        if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = np.array(cv2.imread(os.path.join(testdir, im), cv2.IMREAD_GRAYSCALE), dtype=np.float32) / 255.0

            np.random.seed(seed=0)
            y = x + np.random.normal(0, sigma / 255.0, x.shape)
            y = y.astype(np.float32)
            y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])

            torch.cuda.synchronize()
            y_ = y_.cuda()
            x_ = model(y_)

            x_ = x_.view(y.shape[0], y.shape[1])
            x_ = x_.cpu()
            x_ = x_.detach().numpy().astype(np.float32)

            torch.cuda.synchronize()
            psnr_x_ = compare_psnr(x, x_)

            psnrs.append(psnr_x_)

    psnr_avg = np.mean(psnrs)

    print('* Epoch: [{0}/{1}]\t PSNR {acc:.3f}dB'.format(epoch, args.epochs, acc=psnr_avg))
    return psnr_avg

def train_with_distill(train_loader, d_net, optimizer, criterion_CE, epoch):
    d_net.train()
    d_net.module.s_net.train()
    d_net.module.t_net.train()

    train_loss = AverageMeter()
    acc_psnr = AverageMeter()

    for i, (inputs_low, targets, inputs_high) in enumerate(train_loader):
        targets = targets.cuda(async=True)
        batch_size = inputs_low.shape[0]
        outputs, loss_distill = d_net(inputs_low, inputs_high)

        loss_CE = criterion_CE(outputs, targets)
        loss_distill = loss_distill.sum()/batch_size
        loss = loss_CE + 1e-4*loss_distill

        acc1 = get_psnr(outputs, targets)

        train_loss.update(loss.item(), batch_size)
        acc_psnr.update(acc1, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Train with distillation: [Epoch %d/%d][Batch %d/%d]\t Loss %.3f (L2 %.3f, distill %.3f), PSNR %.3f' %
                  (epoch, args.epochs, i, len(train_loader), train_loss.avg, loss_CE.item(), loss_distill, acc_psnr.avg))


def save_checkpoint(state, is_best, model, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.net_type)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.net_type) + 'model_best.pth.tar')
        torch.save(model, 'runs/%s/'%(args.net_type) + 'model.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def get_psnr(output, target):
    res = []
    for i in range(output.size(0)):
        output_ = output[i].cpu().detach().numpy().astype(np.float32).reshape(output.size(2),output.size(3))
        target_ = target[i].cpu().detach().numpy().astype(np.float32).reshape(output.size(2),output.size(3))
        res.append(compare_psnr(np.clip(output_,0,1), target_))
    return sum(res)/len(res)

if __name__ == '__main__':
    main()
