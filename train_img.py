'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import os
import sys
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets

from convnet_aig import *
import math
from visdom import Visdom
import numpy as np


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lrdecay', default=30, type=int,
                    help='epochs to decay lr')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lrfact', default=1, type=float,
                    help='learning rate factor')
parser.add_argument('--lossfact', default=1, type=float,
                    help='loss factor')
parser.add_argument('--target', default=0.7, type=float, help='target rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='folder path to save checkpoint (default: none)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--expname', default='give_me_a_name', type=str, metavar='n',
                    help='name of experiment (default: test')
parser.set_defaults(test=False)
parser.set_defaults(visdom=False)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    torch.cuda.manual_seed(args.seed)

    if args.visdom:
        global plotter
        plotter = VisdomLinePlotter(env_name=args.expname)
        
    # set the target rates for each layer
    # the default is to use the same target rate for each layer
    target_rates_list = [args.target] * 16
    target_rates = {i:target_rates_list[i] for i in range(len(target_rates_list))}

    model = ResNet50_ImageNet()

    # optionally initialize from pretrained
    if args.pretrained:
        latest_checkpoint = args.pretrained
        if os.path.isfile(latest_checkpoint):
            print("=> loading checkpoint '{}'".format(latest_checkpoint))
            # TODO: clean this part up
            checkpoint = torch.load(latest_checkpoint)
            state = model.state_dict()
            loaded_state_dict = checkpoint
            for k in loaded_state_dict:
                if k in state:
                    state[k] = loaded_state_dict[k]
                else:
                    if 'fc' in k:
                        state[k.replace('fc', 'linear')] = loaded_state_dict[k]
                    if 'downsample' in k:
                        state[k.replace('downsample', 'shortcut')] = loaded_state_dict[k]
            model.load_state_dict(state) 
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(latest_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(latest_checkpoint))
    
    model = torch.nn.DataParallel(model).cuda()

    # ImageNet Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=10, pin_memory=True)


    # optionally resume from a checkpoint
    if args.resume:
        latest_checkpoint = os.path.join(args.resume, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if 'fc' in name],
                            'lr': args.lrfact * args.lr, 'weight_decay': args.weight_decay},
                            {'params': [param for name, param in model.named_parameters() if 'fc' not in name],
                            'lr': args.lr, 'weight_decay': args.weight_decay}
                            ], momentum=args.momentum)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.test:
        test_acc = validate(val_loader, model, criterion, 60, target_rates)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, target_rates)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, target_rates)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch, target_rates):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_c = AverageMeter()
    losses_t = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    activations = AverageMeter()

    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to train mode
    model.train()

    end = time.time()

    ttt = torch.FloatTensor(33).fill_(0)
    ttt = ttt.cuda()
    ttt = torch.autograd.Variable(ttt, requires_grad=False)

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, activation_rates = model(input_var, temperature=temp)

        # classification loss
        loss_classify = criterion(output, target_var)

        # target rate loss
        acts = 0
        acts_plot = 0
        for j, act in enumerate(activation_rates):
            if target_rates[j] < 1:
                acts_plot += torch.mean(act)
                acts += torch.pow(target_rates[j] - torch.mean(act), 2)
            else:
                acts_plot += 1
        
        # this is important when using data DataParallel
        acts_plot = torch.mean(acts_plot / len(activation_rates))
        acts = torch.mean(acts / len(activation_rates))

        act_loss = args.lossfact * acts
        loss = loss_classify + act_loss

        # Sometimes this value is nan 
        # If someone can find out why, please add a pull request
        # For now, we skip the batch and move on
        if math.isnan(acts_plot.data[0]):
            print(activation_rates)
            optimizer.zero_grad()
            loss.backward()
            continue 

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        losses_c.update(loss_classify.data[0], input.size(0))
        losses_t.update(act_loss.data[0], input.size(0))

        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        activations.update(acts_plot.data[0], 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) c({lossc.avg:.4f}) a({lossa.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, lossa=losses_t, lossc=losses_c, top1=top1, top5=top5, act=activations))

    # log values to visdom
    if args.visdom:
        plotter.plot('act', 'train', epoch, activations.avg)
        plotter.plot('top1', 'train', epoch, top1.avg)
        plotter.plot('top5', 'train', epoch, top5.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)


def validate(val_loader, model, criterion, epoch, target_rates):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    accumulator = ActivationAccum_img(epoch)
    activations = AverageMeter()

    # Temperature of Gumble Softmax 
    # We simply keep it fixed
    temp = 1

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, activation_rates = model(input_var, temperature=temp)

        # classification loss
        loss = criterion(output, target_var)

        acts = 0
        for j, act in enumerate(activation_rates):
            if target_rates[j] < 1:
                acts += torch.mean(act)
            else:
                acts += 1
        # this is important when using data DataParallel
        acts = torch.mean(acts / len(activation_rates))
 
        # see above
        if math.isnan(acts.data[0]):
             continue

        # accumulate statistics over eval set
        accumulator.accumulate(activation_rates, target_var, target_rates)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        activations.update(acts.data[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Activations: {act.val:.3f} ({act.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1, top5=top5, act=activations))

    activ_output = accumulator.getoutput()

    print('gate activation rates:')
    print(activ_output[0])

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    if args.visdom:
        plotter.plot('act', 'test', epoch, activations.avg)
        plotter.plot('top1', 'test', epoch, top1.avg)
        plotter.plot('top5', 'test', epoch, top5.avg)
        plotter.plot('loss', 'test', epoch, losses.avg)
        for gate in activ_output[0]:
            plotter.plot('gates', '{}'.format(gate), epoch, activ_output[0][gate])

        # Plot more detailed stats like activation heatmaps for key epochs
        if epoch in [30, 60, 99]:
            for category in activ_output[1]:
                plotter.plot('classes', '{}'.format(category), epoch, activ_output[1][category])
            
            heatmap = activ_output[2]
            means = np.mean(heatmap, axis=0)
            stds = np.std(heatmap, axis=0)
            normalized_stds = np.array(stds / (means + 1e-10)).squeeze()

            plotter.plot_heatmap(activ_output[2], epoch)
            for counter in range(len(normalized_stds)):
                plotter.plot('activations{}'.format(epoch), 'activations', counter, normalized_stds[counter])
            for counter in range(len(means)):
                plotter.plot('opening_rate{}'.format(epoch), 'opening_rate', counter, means[counter])
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.expname) + 'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name], name=split_name)
    def plot_heatmap(self, map, epoch):
        self.viz.heatmap(X=map,
                         env=self.env,
                         opts=dict(title='activations {}'.format(epoch),
                                   xlabel='modules',
                                   ylabel='classes'
                                   ))

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
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lrdecay))
    factor = args.lrfact
    if args.visdom:
        plotter.plot('learning_rate', 'train', epoch, lr)
    optimizer.param_groups[0]['lr'] = factor * lr
    optimizer.param_groups[1]['lr'] = lr

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
    main()
