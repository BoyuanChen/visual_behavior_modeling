


import os
import shutil
import random
from tqdm import tqdm


def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


train_test_split = 0.9


data_folder = 'rgb_data_imgs'
target_folder = 'rgb_target_imgs'

train_data_folder = 'rgb_train_data_imgs'
train_target_folder = 'rgb_train_target_imgs'
test_data_folder = 'rgb_test_data_imgs'
test_target_folder = 'rgb_test_target_imgs'

mkdir(train_data_folder)
mkdir(train_target_folder)
mkdir(test_data_folder)
mkdir(test_target_folder)


file_list = os.listdir(data_folder)
random.shuffle(file_list)
num_files = len(file_list)
train_file_list = file_list[:int(num_files * train_test_split)]
test_file_list = file_list[int(num_files * train_test_split):]

for p_file in tqdm(train_file_list):
    ori_filepath = os.path.join(data_folder, p_file)
    new_filepath = os.path.join(train_data_folder, p_file)
    shutil.copy(ori_filepath, new_filepath)
    ori_filepath = os.path.join(target_folder, p_file)
    new_filepath = os.path.join(train_target_folder, p_file)
    shutil.copy(ori_filepath, new_filepath)


for p_file in tqdm(test_file_list):
    ori_filepath = os.path.join(data_folder, p_file)
    new_filepath = os.path.join(test_data_folder, p_file)
    shutil.copy(ori_filepath, new_filepath)
    ori_filepath = os.path.join(target_folder, p_file)
    new_filepath = os.path.join(test_target_folder, p_file)
    shutil.copy(ori_filepath, new_filepath)