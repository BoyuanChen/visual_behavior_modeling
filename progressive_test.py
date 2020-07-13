
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
from tqdm import tqdm
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

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# from models.cnned import SegNet
from new_models.model import TrajPredictor

# model = SegNet(n_classes=args.n_classes, is_unpooling=True)
model = TrajPredictor()

if args.cuda:
    model = torch.nn.DataParallel(model).cuda()
if args.cuda:
    ckpt = torch.load('./checkpoint_half/checkpoint.pth.tar')
    model.load_state_dict(ckpt['state_dict'])
else:
    ckpt = torch.load('./checkpoint_half/checkpoint.pth.tar', map_location=lambda storage, loc: storage)
    from collections import OrderedDict 
    cpu_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        name = k[7:] # remove `module.`
        cpu_state_dict[name] = v
    model.load_state_dict(cpu_state_dict)

lossfun = nn.MSELoss()


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


policy = 'single_obs_1_food' 
test_folder = 'data_arr_progressive_' + policy
test_files = os.listdir(test_folder)
for p_file in tqdm(test_files):
    filepath = os.path.join(test_folder, p_file)
    num_frames = len(os.listdir(filepath))
    for frame_idx in range(num_frames):
        arr_filename = 'frame_' + str(frame_idx) + '.npy'
        arr_filepath = os.path.join(filepath, arr_filename)
        arr = np.load(arr_filepath)

        arr = arr / 255.0
        arr = torch.tensor(arr)
        arr = arr.unsqueeze(0)
        arr = arr.permute(0, 3, 1, 2)
        if args.cuda:
            arr = arr.cuda()
        arr = Variable(arr)
        output = model(arr.float())

        output = output[0].permute(1, 2, 0).data.cpu().numpy()
        output = (output * 255.0).astype('uint8')

        output_filename = 'output_' + str(frame_idx) + '.npy'
        output_filepath = os.path.join(filepath, output_filename)
        np.save(output_filepath, output)