from __future__ import print_function
import time
import shutil
import pathlib

import os
from os.path import isfile, join

import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from mobilenet import MobileNet
from config import cfg


best_prec1 = 0


def main():
    global opt, start_epoch, best_prec1
    opt = cfg
    opt.gpuids = list(map(int, opt.gpuids))

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    model = MobileNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
            momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    start_epoch = 0

    ckpt_file = join("model", opt.ckpt)

    if opt.cuda:
        torch.cuda.set_device(opt.gpuids[0])
        with torch.cuda.device(opt.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=opt.gpuids,
                                output_device=opt.gpuids[0])
        cudnn.benchmark = True

    # for resuming training
    if opt.resume:
        if isfile(ckpt_file):
            print("==> Loading Checkpoint '{}'".format(opt.ckpt))
            if opt.cuda:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(opt.gpuids[0]))
                try:
                    model.module.load_state_dict(checkpoint['model'])
                except:
                    model.load_state_dict(checkpoint['model'])
            else:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)

                try:
                    model.load_state_dict(checkpoint['model'])
                except:
                    # create new OrderedDict that does not contain `module.`
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model'].items():
                        if k[:7] == 'module.':
                            name = k[7:] # remove `module.`
                        else:
                            name = k[:]
                        new_state_dict[name] = v

                    model.load_state_dict(new_state_dict)
            
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("==> Loaded Checkpoint '{}' (epoch {})".format(
                        opt.ckpt, start_epoch))
        else:
            print("==> no checkpoint found at '{}'".format(
                        opt.ckpt))
            return

    # Download & Load Dataset
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)

    valset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_val)
    val_loader = torch.utils.data.DataLoader(
            valset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # for evaluation
    if opt.eval:
        if isfile(ckpt_file):
            print("==> Loading Checkpoint '{}'".format(opt.ckpt))
            if opt.cuda:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(opt.gpuids[0]))
                try:
                    model.module.load_state_dict(checkpoint['model'])
                except:
                    model.load_state_dict(checkpoint['model'])
            else:
                checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)

                try:
                    model.load_state_dict(checkpoint['model'])
                except:
                    # create new OrderedDict that does not contain `module.`
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['model'].items():
                        if k[:7] == 'module.':
                            name = k[7:] # remove `module.`
                        else:
                            name = k[:]
                        new_state_dict[name] = v

                    model.load_state_dict(new_state_dict)
            
            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("==> Loaded Checkpoint '{}' (epoch {})".format(
                        opt.ckpt, start_epoch))

            # evaluate on validation set
            print("\n===> [ Evaluation ]")
            start_time = time.time()
            prec1 = validate(val_loader, model, criterion)
            elapsed_time = time.time() - start_time
            print("====> {:.2f} seconds to evaluate this model\n".format(
                elapsed_time))
            return
        else:
            print("==> no checkpoint found at '{}'".format(
                        opt.ckpt))
            return

    # train...
    train_time = 0.0
    validate_time = 0.0
    for epoch in range(start_epoch, opt.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\n==> Epoch: {}, lr = {}'.format(epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        print("===> [ Training ]")
        start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print("====> {:.2f} seconds to train this epoch\n".format(
                    elapsed_time))
        
        # evaluate on validation set
        print("===> [ Validation ]")
        start_time = time.time()
        prec1 = validate(val_loader, model, criterion)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print("====> {:.2f} seconds to validate this epoch\n".format(
            elapsed_time))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_model(state, epoch, is_best)
    
    avg_train_time = train_time/opt.epochs
    avg_valid_time = validate_time/opt.epochs
    total_train_time = train_time+validate_time
    print("====> average training time per epoch: {}m {:.2f}s".format(int(avg_train_time//60), avg_train_time%60))
    print("====> average validation time per epoch: {}m {:.2f}s".format(int(avg_valid_time//60), avg_valid_time%60))
    print("====> training time: {}m {:.2f}s".format(int(train_time//60), train_time%60))
    print("====> validation time: {}m {:.2f}s".format(int(validate_time//60), validate_time%60))
    print("====> total training time: {}m {:.2f}s".format(int(total_train_time//60), total_train_time%60))


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for _, (input, target) in enumerate(tqdm(train_loader, dynamic_ncols=True, unit='batch')):
        if opt.cuda:
            target = target.cuda(async=True)
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(' * Training Prec@1 {top1.avg:.3f}\t Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for _, (input, target) in enumerate(tqdm(val_loader, dynamic_ncols=True, unit='batch')):
        if opt.cuda:
            target = target.cuda(async=True)
        
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print(' * Prec@1 {top1.avg:.3f}\t Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_model(state, epoch, is_best):
    dir_path = "model"
    pathlib.Path(dir_path).mkdir(exist_ok=True)

    model_file = join(dir_path, "ckpt_epoch_{}.pth".format(epoch))

    if is_best:
        shutil.rmtree("model")
        pathlib.Path(dir_path).mkdir(exist_ok=True)

        torch.save(state, model_file)
        shutil.copyfile(model_file, 'model/ckpt_best.pth')
    else:
        torch.save(state, model_file)


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
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 50))
    # lr = opt.lr * (0.98 ** epoch)
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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("====> total time: {}m {:.2f}s".format(elapsed_time//60, elapsed_time%60))
