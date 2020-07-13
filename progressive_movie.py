


import os
import sys
import yaml
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from subprocess import call



def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


policy = 'single_obs_1_food'
output_video_folder = 'output_progressive_' + policy 
test_folder = 'data_arr_progressive_' + policy
test_files = os.listdir(test_folder)
mkdir(output_video_folder)
for p_file in tqdm(test_files):
    filepath = os.path.join(test_folder, p_file)
    output_video_filepath = os.path.join(output_video_folder, p_file)
    mkdir(output_video_filepath)
    num_frames = int(len(os.listdir(filepath)) / 2)
    for frame_idx in range(num_frames):
        # load input and output
        input_filename = 'frame_' + str(frame_idx) + '.npy'
        input_filepath = os.path.join(filepath, input_filename)
        output_filename = 'output_' + str(frame_idx) + '.npy'
        output_filepath = os.path.join(filepath, output_filename)
        input_arr = np.load(input_filepath)
        output_arr = np.load(output_filepath)

        input_img = Image.fromarray(input_arr.astype('uint8'))
        output_img = Image.fromarray(output_arr.astype('uint8'))

        input_img.save(os.path.join(output_video_filepath, 'input_' + str(frame_idx) + '.png'))
        output_img.save(os.path.join(output_video_filepath, 'output_' + str(frame_idx) + '.png'))

    i_command = "ffmpeg -r 30 -f image2 -s 1920x1080 -i " + output_video_filepath + "/input_%1d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p " + output_video_filepath + "/input_video.mp4" + " -loglevel quiet"
    o_command = "ffmpeg -r 30 -f image2 -s 1920x1080 -i " + output_video_filepath + "/output_%1d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p " + output_video_filepath + "/output_video.mp4" + " -loglevel quiet"
    exit_code = call(i_command, shell=True)
    exit_code = call(o_command, shell=True)

