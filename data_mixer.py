


import os
import shutil
import random
import numpy as np

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


if __name__ == '__main__':

    filebase = 'data_half'
    filelist = ['final_data_straight',
                'final_data_bezier3',
                'final_data_bezier4',
                'final_data_single_obs_1_food',
                'final_data_single_obs_2_foods_train_half_half_test_normal']
    filelist = [os.path.join(filebase, x) for x in filelist]
    random.seed(100)

    final_data_folder = 'final_data'

    mkdir(final_data_folder)

    train_data = []
    train_target = []
    train_policy_ids = []
    test_data = []
    test_target = []
    test_policy_ids = []

    p_idx = 0
    for p_policy in filelist:
        train_data.append(np.load(os.path.join(p_policy, 'train_data.npy')))
        train_target.append(np.load(os.path.join(p_policy, 'train_target.npy')))
        # get the policy ids
        ids_arr = [p_idx] * train_data[-1].shape[0]
        ids_arr = np.expand_dims(ids_arr, axis=1)
        train_policy_ids.append(ids_arr)
        test_data.append(np.load(os.path.join(p_policy, 'test_data.npy')))
        test_target.append(np.load(os.path.join(p_policy, 'test_target.npy')))
        ids_arr = [p_idx] * test_data[-1].shape[0]
        ids_arr = np.expand_dims(ids_arr, axis=1)
        test_policy_ids.append(ids_arr)
        p_idx = p_idx + 1

    train_data = np.concatenate(train_data, axis=0)
    train_target = np.concatenate(train_target, axis=0)
    train_policy_ids = np.concatenate(train_policy_ids, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    test_target = np.concatenate(test_target, axis=0)
    test_policy_ids = np.concatenate(test_policy_ids, axis=0)

    num_train_data = train_data.shape[0]
    train_ids = np.arange(0, num_train_data)
    random.shuffle(train_ids)
    train_data = train_data[train_ids]
    train_target = train_target[train_ids]
    train_policy_ids = train_policy_ids[train_ids]

    num_test_data = test_data.shape[0]
    test_ids = np.arange(0, num_test_data)
    random.shuffle(test_ids)
    test_data = test_data[test_ids]
    test_target = test_target[test_ids]
    test_policy_ids = test_policy_ids[test_ids]

    np.save(os.path.join(final_data_folder, 'train_data.npy'), train_data)
    np.save(os.path.join(final_data_folder, 'train_target.npy'), train_target)
    np.save(os.path.join(final_data_folder, 'train_policy_ids.npy'), train_policy_ids)
    np.save(os.path.join(final_data_folder, 'test_data.npy'), test_data)
    np.save(os.path.join(final_data_folder, 'test_target.npy'), test_target)
    np.save(os.path.join(final_data_folder, 'test_policy_ids.npy'), test_policy_ids)



