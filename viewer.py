
"""
This code is for plotting real size output image. For high resolution images for demonstration and paper, please refer to
demonstration_viewer.py
"""


import os
import sys
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm


test_results_folder = './test_results'
test_results_img_folder = './test_results/images'

test_results_files = os.listdir(test_results_folder)
if os.path.exists(test_results_img_folder):
    shutil.rmtree(test_results_img_folder)
os.makedirs(test_results_img_folder)


height = 64
width = 192

test_results_files = ['test_resutls_90.npy']

for p_file in tqdm(test_results_files):
    p_file_path = os.path.join(test_results_folder, p_file)
    p_epoch_res = np.load(p_file_path)
    epoch_idx = p_file.split('.')[0].split('_')[2]
    epoch_folder = 'epoch_' + epoch_idx
    epoch_folder = os.path.join(test_results_img_folder, epoch_folder)
    os.makedirs(epoch_folder)

    index = 0
    num_batch = p_epoch_res.shape[0]
    for p_batch in tqdm(range(num_batch)):
        data = p_epoch_res[p_batch][0]
        tar = p_epoch_res[p_batch][1]
        res = p_epoch_res[p_batch][2]
        batch_size = tar.shape[0]
        for p_data in range(batch_size):
            # get output image
            out_img = res[p_data]
            out_img = np.transpose(out_img, (1, 2, 0))
            # out_img = out_img * 128 + 128
            out_img = out_img * 255
            # import IPython
            # IPython.embed()
            # assert False
            out_img = Image.fromarray(out_img.astype('uint8'))
            # get target image
            tar_out_img = tar[p_data]
            tar_out_img = np.transpose(tar_out_img, (1, 2, 0))
            # tar_out_img = tar_out_img * 128 + 128
            tar_out_img = tar_out_img * 255
            tar_out_img = Image.fromarray(tar_out_img.astype('uint8'))
            # get data image
            data_out_img = data[p_data]
            data_out_img = np.transpose(data_out_img, (1, 2, 0))
            # data_out_img = data_out_img * 128 + 128
            data_out_img = data_out_img * 255
            data_out_img = Image.fromarray(data_out_img.astype('uint8'))
            # save them side by side
            new = Image.new('RGB', (width, height))
            new.paste(data_out_img, (0, 0))
            new.paste(tar_out_img, (64, 0))
            new.paste(out_img, (128, 0))
            filename = 'img_' + str(index) + '.png'
            filepath = os.path.join(epoch_folder, filename)
            new.save(filepath)
            index += 1