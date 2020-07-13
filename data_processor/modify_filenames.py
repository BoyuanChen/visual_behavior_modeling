

# import os
# from tqdm import tqdm
# from subprocess import call

# base_path = './data_imgs/rgb_target_imgs'
# filelist = os.listdir(base_path)

# for p_file in tqdm(filelist):
#   filepath = os.path.join(base_path, p_file)
#   new_file_name = 'cur_' + p_file.split('_')[1]
#   new_file_name = os.path.join(base_path, new_file_name)
#   subprocess_command_line = 'mv ' + str(filepath) + ' ' + str(new_file_name)
#   call(subprocess_command_line, shell=True)








import os
import shutil
from tqdm import tqdm
from subprocess import call

base_path = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/data_imgs/rgb_train_data_imgs/output'
saved_path = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/augmented_train_data_imgs'
filelist = os.listdir(base_path)



for p_file in tqdm(filelist):
    pre = p_file.split('_')[1]
    if pre == 'groundtruth':
        ori_target_name = os.path.join(base_path, p_file)
        ori_data_name = '_'.join(p_file.split('_')[3:])
        ori_data_name = '_'.join(ori_data_name.split('_')[:4] + ['original'] + ori_data_name.split('_')[4:])
        ori_data_name = os.path.join(base_path, ori_data_name)


        target_name = 'end_' + p_file.split('_')[8].split('.')[0] + '_' + p_file.split('_')[9]
        data_name = 'cur_' + p_file.split('_')[8].split('.')[0] + '_' + p_file.split('_')[9]

        target_name = os.path.join(saved_path, target_name)
        data_name = os.path.join(saved_path, data_name)

        shutil.copy(ori_target_name, target_name)
        shutil.copy(ori_data_name, data_name)



base_path = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/data_imgs/rgb_test_data_imgs/output'
saved_path = '/home/cml/bo/ToM_Base/sim_tom/rgb/tom_simple_rgb/data_processor/augmented_test_data_imgs'
filelist = os.listdir(base_path)



for p_file in tqdm(filelist):
    pre = p_file.split('_')[1]
    if pre == 'groundtruth':
        ori_target_name = os.path.join(base_path, p_file)
        ori_data_name = '_'.join(p_file.split('_')[3:])
        ori_data_name = '_'.join(ori_data_name.split('_')[:4] + ['original'] + ori_data_name.split('_')[4:])
        ori_data_name = os.path.join(base_path, ori_data_name)


        target_name = 'end_' + p_file.split('_')[8].split('.')[0] + '_' + p_file.split('_')[9]
        data_name = 'cur_' + p_file.split('_')[8].split('.')[0] + '_' + p_file.split('_')[9]

        target_name = os.path.join(saved_path, target_name)
        data_name = os.path.join(saved_path, data_name)

        shutil.copy(ori_target_name, target_name)
        shutil.copy(ori_data_name, data_name)