


import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

data_filepath = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/special_test_data_imgs/rgb_data_imgs'
target_filepath = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/special_test_data_imgs/rgb_target_imgs'
data_arr_filepath = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/special_test_data_imgs/rgb_data_arr'
target_arr_filepath = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/special_test_data_imgs/rgb_target_arr'


mkdir(data_arr_filepath)
mkdir(target_arr_filepath)

filelist = os.listdir(data_filepath)


for p_file in tqdm(filelist):
    im = Image.open(os.path.join(data_filepath, p_file))
    np_im = np.array(im)
    np.save(os.path.join(data_arr_filepath, p_file.split('.')[0] + '.npy'), np_im)





filelist = os.listdir(target_filepath)


for p_file in tqdm(filelist):
    im = Image.open(os.path.join(target_filepath, p_file))
    np_im = np.array(im)
    np.save(os.path.join(target_arr_filepath, p_file.split('.')[0] + '.npy'), np_im)