
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import time
import shutil
import itertools
import numpy as np
from utils import Bar, Logger, AverageMeter, regression_accuracy, mkdir_p, savefig

# Training settings
parser = argparse.ArgumentParser(description='Observer Network')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 30, 50, 80],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--n-classes', default=3, type=int, metavar='N',
                    help='number of classes')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

best_acc = float("-inf")

if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# from models.cnned import SegNet
from new_models.model import TrajPredictor

# model = SegNet(n_classes=args.n_classes, is_unpooling=True)
model = TrajPredictor()


if args.cuda:
    model = torch.nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
lossfun = nn.MSELoss()

logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title='observer')
logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

test_results_path = './test_results'
mkdir(test_results_path)

train_data = np.load('./final_data/train_data.npy')
train_target = np.load('./final_data/train_target.npy')
train_policy_ids = np.load('./final_data/train_policy_ids.npy')
test_data = np.load('./final_data/test_data.npy')
test_target = np.load('./final_data/test_target.npy')
test_policy_ids = np.load('./final_data/test_policy_ids.npy')

# train_data = (train_data - 128) / 128
# train_target = (train_target - 128) / 128
# test_data = (test_data - 128) / 128
# test_target = (test_target - 128) / 128


train_data = train_data / 255.0
train_target = train_target / 255.0
test_data = test_data / 255.0
test_target = test_target / 255.0


train_data = torch.FloatTensor(train_data)
train_target = torch.FloatTensor(train_target)
train_policy_ids = torch.FloatTensor(train_policy_ids)
test_data = torch.FloatTensor(test_data)
test_target = torch.FloatTensor(test_target)
test_policy_ids = torch.FloatTensor(test_policy_ids)

final_train_data = torch.utils.data.TensorDataset(train_data, train_target, train_policy_ids)
final_test_data = torch.utils.data.TensorDataset(test_data, test_target, test_policy_ids)

trainloader = torch.utils.data.DataLoader(final_train_data, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
testloader = torch.utils.data.DataLoader(final_test_data, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

def train():
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    scores = []
    for batch_idx, (data, target, policy_ids) in enumerate(trainloader):

        data = data.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)


        data_time.update(time.time() - end)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = lossfun(output, target)

        # measure accuracy and record loss
        prec1 = 0.0
        losses.update(loss.item(), data.size(0))
        top1.update(prec1, data.size(0))

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top1.avg,
                    )
        bar.next()

    return (losses.avg, -top1.avg)

def test(save_flag, epoch):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    if save_flag == True:
        saved_test_results = []
        saved_test_results_policy_ids = []

    for batch_idx, (data, target, policy_ids) in enumerate(testloader):

        data = data.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = lossfun(output, target)

        prec1 = 0.0
        losses.update(loss.item(), data.size(0))
        top1.update(prec1, data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top1.avg,
                    )
        bar.next()

        # save outputs
        if save_flag == True:
            output = output.cpu().detach().numpy() # (100, 12288)
            target = target.cpu().numpy()
            data = data.cpu().numpy()
            policy_ids = policy_ids.cpu().numpy()
            saved_test_results.append([data, target, output])
            saved_test_results_policy_ids.append([policy_ids])

    if save_flag == True:
        saved_test_results = np.array(saved_test_results)
        saved_test_results_policy_ids = np.array(saved_test_results_policy_ids)
        filename = "test_resutls_" + str(epoch) + ".npy"
        filepath = os.path.join(test_results_path, filename)
        np.save(filepath, saved_test_results)
        filename = "test_resutls_policy_ids_" + str(epoch) + ".npy"
        filepath = os.path.join(test_results_path, filename)
        np.save(filepath, saved_test_results_policy_ids)


    bar.finish()

    return (losses.avg, -top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

for epoch in range(0, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    if epoch % 10 == 0:
        save_flag = True
    else:
        save_flag = False

    train_loss, train_acc = train()
    test_loss, test_acc = test(save_flag, epoch)
    logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)

    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

logger.close()
print('Best acc: ', best_acc)