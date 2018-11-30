from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import shutil
import time
import math
import warnings
# import models
# import random
import cfar_10_data as cfar

parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--no-save-model', dest='no_save_model', action='store_true', default=False,
                    help='only save best model (default: false)')
parser.add_argument('--all-random', dest='all_random', default=True,
                    help='only save best model (default: false)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='manual seed (default: 0)')
parser.add_argument('--gpu', default='5',
                    help='gpu available')
parser.add_argument('--range', default='1-5', type=str,
                    help='range_num')
parser.add_argument('--savedir', type=str, metavar='PATH', default='results/savedir',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--resume', action='store_true',
                    help='use latest checkpoint if have any (default: none)')

parser.add_argument('--stages', type=str, metavar='STAGE DEPTH', default='3-3-3',
                    help='per layer depth')
parser.add_argument('--bottleneck', default=2, type=int, metavar='B',
                    help='bottleneck (default: 4)')
parser.add_argument('--group-1x1', type=int, metavar='G', default=1,
                    help='1x1 group convolution (default: 1)')
parser.add_argument('--group-3x3', type=int, metavar='G', default=1,
                    help='3x3 group convolution (default: 1)')
parser.add_argument('--condense-factor', type=int, metavar='C', default=1,
                    help='condense factor (default: 4)')
parser.add_argument('--growth', type=str, metavar='GROWTH RATE', default='8-8-8',
                    help='per layer growth')
parser.add_argument('--reduction', default=0.5, type=float, metavar='R',
                    help='transition reduction (default: 0.5)')
parser.add_argument('--dropout-rate', default=0, type=float,
                    help='drop out (default: 0)')
parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
                    help='group lasso loss weight (default: 0)')

parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')
parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--model', default='resnet', type=str, metavar='M',
                    help='model to train the dataset')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.stages = list(map(int, args.stages.split('-')))
args.growth = list(map(int, args.growth.split('-')))
if args.condense_factor is None:
    args.condense_factor = args.group_1x1
# args.data == 'cifar10'
if args.data == 'cifar10':
    args.num_classes = 10
    # args.model = 'resnet'
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

warnings.filterwarnings("ignore")

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import copy
import os
import operator
from functools import reduce
from cells import CellResNet, CellNAS
import hashlib
import collections
import json
import random
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

best_prec1 = 100
class ArcResNet(nn.Module):
    def __init__(self,
                 init_channel,
                 init_ks,
                 n_class,
                 num_layer_list,
                 out_channel_list2D,
                 kernel_size_list2D,
                 stride_list,
                 skip_connect_list,
                 skip_kernel_size_list,
                 **kwargs):
        """
        :param init_channel: type int, channels of initial convolution layer
        :param init_ks: type int, kernel size of initial convolution layer
        :param n_class: type int, number of class
        :param num_layer_list: type 1D list, how many layers in each cell
        :param out_channel_list2D: type 2D list, out channels for each cell
        :param kernel_size_list2D: type 2D list, kernel size for each cell
        :param stride_list: type 1D list, stride for each cell
        :param skip_connect_list: type 1D list,
        :param skip_kernel_size_list: type 1D list
        """
        super(ArcResNet, self).__init__()
        assert len(num_layer_list) == len(out_channel_list2D), 'length of num_layer_list should equal ' \
                                                                'to length of out channels_list2D'
        self.init_channel = init_channel
        self.init_ks = init_ks
        self.n_class = n_class
        self.num_layer_list = num_layer_list
        self.out_channel_list2D = out_channel_list2D
        self.kernel_size_list2D = kernel_size_list2D
        self.stride_list = stride_list
        self.skip_connect_list = skip_connect_list
        self.skip_kernel_size_list = skip_kernel_size_list

        # build network
        self.init_conv = nn.Conv2d(3,
                                   self.init_channel,
                                   kernel_size=self.init_ks,
                                   padding=self.init_ks // 2,
                                   bias=False)
        self.init_bn = nn.BatchNorm2d(self.init_channel)
        self.layers = nn.Sequential()
        in_channel = self.init_channel
        for i, n in enumerate(self.num_layer_list):
            self.layers.add_module('cell_%d' % i, CellResNet(num_layer=num_layer_list[i],
                                                             in_channel=in_channel,
                                                             out_channel_list=out_channel_list2D[i],
                                                             kernel_size_list=kernel_size_list2D[i],
                                                             stride=stride_list[i],
                                                             skip_connect=skip_connect_list[i],
                                                             skip_kernel_size=skip_kernel_size_list[i]))
            in_channel = out_channel_list2D[i][-1]
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channel, self.n_class)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.layers(out)
        out = self.global_pooling(out)
        out = self.fc(out.view(out.size(0), -1))
        return out

    def as_dict(self):
        d = collections.OrderedDict()
        d['init_channel'] = self.init_channel
        d['init_ks'] = self.init_ks
        d['n_class'] = self.n_class
        d['num_layer_list'] = self.num_layer_list
        d['out_channel_list2D'] = self.out_channel_list2D
        d['kernel_size_list2D'] = self.kernel_size_list2D
        d['stride_list'] = self.stride_list
        d['skip_connect_list'] = self.skip_connect_list
        d['skip_kernel_size_list'] = self.skip_kernel_size_list
        return d

    def save_param(self, dir_path='./'):
        with open(os.path.join(dir_path, 'params.json'), 'w') as f:
            json.dump(self.as_dict(), f)

    def load_param(self, dir_path='./'):
        assert os.path.exists(os.path.join(dir_path, 'params.json'))==True
        with open(os.path.join(dir_path, 'params.json'), 'r') as f:
            d = json.load(f)
        self.__init__(**d)

    @property
    def md5name(self):
        return hashlib.md5(json.dumps(self.as_dict()).encode('utf-8')).hexdigest()

    def count_params(self):
        n_params = sum([reduce(operator.mul, i.size(), 1) for i in self.parameters()]) / 1e6
        print('#Param of Model: %.4f M' % n_params)
        return n_params

    def copy(self):
        d = copy.deepcopy(self.as_dict())
        return ArcResNet(**d)
def add_model_s(index, model):
    global args
    os.makedirs(args.savedir, exist_ok=True)
    names = os.listdir(args.savedir)
    exist = False
    # s_finish = False
    # finish = False
    # arch = None
    filename =None
    for name in names:
        if name[-3:] == 'txt'and name.split('_')[0] == str(index):
            exist = True
            filename = name
            break
    if exist:
        filepath = os.path.join(args.savedir, filename)
        fp = open(filepath, "r")
        lines = fp.readlines()
        if len(lines)==args.epochs:
            print('exist but finish')
            return True, model
        else:
            model_dir = os.path.join(args.savedir, filename[:-4])
            model.load_param(dir_path=model_dir)
            # elements = filename[:-4].split('_')[1:]
            # stages = elements[0].split('-')
            # growth = elements[1].split('-')
            # bottleneck = elements[2].split('-')
            # # N = len(stages)
            # args.stages = [int(st) for st in stages]
            # args.growth = [int(gr) for gr in growth]
            # args.bottleneck = [float(bo) for bo in bottleneck]
            print('exist and continue')
            return False, model
    else:
        print('not exist')
        return False, model

def main(index):
    global args, best_prec1
    best_prec1 = 0
    args.start_epoch=0
    if args.all_random:
        times = random.randint(2, 8)
        num_layer_list = [random.randint(1, 4) for gg in range(times)]
        stride_list = [random.randint(1, 2) for gg in range(times)]
        while stride_list.count(2)>3:
            stride_list = [random.randint(1, 2) for gg in range(times)]
        ResNet18_ArcParams = dict(init_channel=random.randint(1, 6) * 8,
                                  init_ks= random.choice([1, 3, 5]),
                                  n_class=10,
                                  num_layer_list=num_layer_list,
                                  out_channel_list2D=[[random.randint(1, 12) * 8 for i in range(num_layer_list[gg])]
                                                      for gg in range(times)],
                                  kernel_size_list2D=[[random.choice([1, 3, 5]) for i in range(num_layer_list[gg])]
                                                      for gg in range(times)],
                                  stride_list=stride_list,
                                  skip_connect_list=[random.choice([True, False]) for gg in range(times)],
                                  skip_kernel_size_list=[random.choice([None, 1]) for gg in range(times)])
    model = ArcResNet(**ResNet18_ArcParams)
    finish, model = add_model_s(index, model)
    if finish:
        return
    args.filename = ('%d_' % index) + model.md5name + '.txt'
    model_dir = os.path.join(args.savedir, args.filename[:-4])
    os.makedirs(model_dir, exist_ok=True)
    model.save_param(dir_path=model_dir)
    print('<<<<=============================>>>>')
    print(args.filename)
    print('<<<<=============================>>>>')
    # model = getattr(models, args.model)(args)
    print(model)
    arch_para = model.as_dict()
    if args.data in ['cifar10', 'cifar100']:
        IMAGE_SIZE = 32
    else:
        IMAGE_SIZE = 224

    ### Create model
    # model = getattr(models, args.model)(args)

    model = torch.nn.DataParallel(model).cuda()

    ### Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    ### Optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])


    cudnn.benchmark = True

    ### Data loading
    if args.data == "cifar10":
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        # train_set = datasets.CIFAR10('../data', train=True, download=True,
        #                              transform=transforms.Compose([
        #                                  transforms.RandomCrop(32, padding=4),
        #                                  transforms.RandomHorizontalFlip(),
        #                                  transforms.ToTensor(),
        #                                  normalize,
        #                              ]))
        train_set = cfar.CIFAR10('./data', train=True, num_valid=5000, valid=False,
                                 transform=transforms.Compose([
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ]))
        test_set = datasets.CIFAR10('./data', train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
        val_set = cfar.CIFAR10('./data', train=True, num_valid=5000, valid=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   normalize,
                               ]))

    elif args.data == "cifar100":
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100('../data', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))
        val_set = datasets.CIFAR100('../data', train=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize,
                                   ]))
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        print(arch_para)
        tr_prec1, tr_prec5, loss, lr = \
            train(train_loader, model, criterion, optimizer, epoch)

        ### Evaluate on validation set
        val_prec1, val_prec5 = validate(val_loader, model, criterion)
        test_prec1, test_prec5 = validate(test_loader, model, criterion)

        ### Remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        # print(is_best, val_prec1, best_prec1)
        best_prec1 = max(val_prec1, best_prec1)
        model_filename = 'lastest.pth.tar'
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" %
                                          (test_prec1, test_prec5, val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr))

    return


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    learned_module_list = []

    ### Switch to train mode
    model.train()
    ### Find all learned convs to prepare for group lasso loss
    for m in model.modules():
        if m.__str__().startswith('LearnedGroupConv'):
            learned_module_list.append(m)
    running_lr = None

    end = time.time()
    start = time.time()
    for i, (input, target) in enumerate(train_loader):
        progress = float(epoch * len(train_loader) + i) / \
            (args.epochs * len(train_loader))
        args.progress = progress
        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)
        if running_lr is None:
            running_lr = lr

        ### Measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        ### Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        ### Add group lasso loss
        if args.group_lasso_lambda > 0:
            lasso_loss = 0
            for m in learned_module_list:
                lasso_loss = lasso_loss + m.lasso_loss
            loss = loss + args.group_lasso_lambda * lasso_loss

        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                  'lr {lr: .4f}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
    print('epoch_time %.4f' % (time.time() - start))
    return top1.avg, top5.avg, losses.avg, running_lr


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ### Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        ### Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, args.filename[:-4])
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.savedir, args.filename)
    model_dir = os.path.join(args.savedir, args.filename[:-4])
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if args.no_save_model:
        shutil.move(model_filename, best_filename)
    elif is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


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


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    begin = int(args.range.split('-')[0])
    end = int(args.range.split('-')[1]) + 1
    for index in range(begin, end):
        # torch.manual_seed(args.manual_seed)
        # torch.cuda.manual_seed_all(args.manual_seed)
        main(index)