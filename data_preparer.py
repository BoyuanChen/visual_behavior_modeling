
import os
import sys
import shutil
import numpy as np
from tqdm import tqdm

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


train_test_split = 0.9

train_data_folder = 'rgb_train_data_arr'
train_target_folder = 'rgb_train_target_arr'
test_data_folder = 'rgb_test_data_arr'
test_target_folder = 'rgb_test_target_arr'

final_data_folder = 'final_data'

mkdir(final_data_folder)



train_data_files = os.listdir(train_data_folder)
train_ids = ['_'.join(x.split('.')[0].split('_')[1:]) for x in train_data_files]
test_data_files = os.listdir(test_data_folder)
test_ids = ['_'.join(x.split('.')[0].split('_')[1:]) for x in test_data_files]


# data_files = os.listdir(data_folder)
# file_ids = ['_'.join(x.split('.')[0].split('_')[1:]) for x in data_files]
# split = int(len(file_ids) * train_test_split)
# train_ids = file_ids[:split]
# test_ids = file_ids[split:]

train_data = []
train_target = []
for idx in tqdm(train_ids):
    data_filename = 'cur_' + str(idx) + '.npy'
    target_filename = 'end_' + str(idx) + '.npy'
    data_filepath = os.path.join(train_data_folder, data_filename)
    target_filepath = os.path.join(train_target_folder, target_filename)
    #load the data and target
    train_data.append(np.load(data_filepath))
    train_target.append(np.load(target_filepath))

train_data = np.array(train_data)
train_target = np.array(train_target)
np.save(os.path.join(final_data_folder, 'train_data'), train_data)
np.save(os.path.join(final_data_folder, 'train_target'), train_target)


test_data = []
test_target = []
test_vis_ids = []
for idx in tqdm(test_ids):
    data_filename = 'cur_' + str(idx) + '.npy'
    target_filename = 'end_' + str(idx) + '.npy'
    data_filepath = os.path.join(test_data_folder, data_filename)
    target_filepath = os.path.join(test_target_folder, target_filename)
    #load the data and target
    test_data.append(np.load(data_filepath))
    test_target.append(np.load(target_filepath))
    test_vis_ids.append(idx)

test_data = np.array(test_data)
test_target = np.array(test_target)
test_vis_ids = np.array(test_vis_ids)
np.save(os.path.join(final_data_folder, 'test_data.npy'), test_data)
np.save(os.path.join(final_data_folder, 'test_target.npy'), test_target)
np.save(os.path.join(final_data_folder, 'test_ids.npy'), test_vis_ids)



