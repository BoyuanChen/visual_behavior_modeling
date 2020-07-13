
"""
One obs and one food:

1. Get the position of the green ball and the robot (black ball) in the input image
2. Get the position of the robot (black ball) in the target image
3. Get the position of the robot (black ball) in the predicted image:
    - If it changes more than 1/2 (starts with nothing) of the radius of the robot:
        - if the closest point on the contour to the target green ball is smaller than the diameter of ther robot, it works.
        - else: no   
    - Else:
        - if the chagnes of the green ball is smaller than the 1/2 (starts with nothing) of the radius of the target green ball. it works
        - else: no
"""


import os
import cv2
import sys
import shutil
import imutils
import numpy as np
from PIL import Image
from tqdm import tqdm

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


test_results_folder = './test_results'
example_measure_imgs_path = './test_results/example_measure_imgs_obs_one_food_pos'
mkdir(example_measure_imgs_path)
redLower = (0, 106, 46)
redUpper = (46, 255, 255)
greenLower = (41, 54, 110)
greenUpper = (74, 255, 255)
blackLower = (0, 0, 0)
blackUpper = (255, 255, 88)
top_k_cnts = 2

test_results_files = os.listdir(test_results_folder)

test_results_files = ['test_resutls_90.npy']
test_results_policy_ids_files = ['test_resutls_policy_ids_90.npy']

for p_file in tqdm(test_results_files):
    p_ids_file_path = os.path.join(test_results_folder, test_results_policy_ids_files[0])
    p_file_path = os.path.join(test_results_folder, p_file)
    p_epoch_res = np.load(p_file_path)
    p_epoch_ids = np.load(p_ids_file_path)

    index = 0
    num_batch = p_epoch_res.shape[0]
    result = []
    for p_batch in tqdm(range(num_batch)):
        data = p_epoch_res[p_batch][0]
        tar = p_epoch_res[p_batch][1]
        res = p_epoch_res[p_batch][2]
        ids = p_epoch_ids[p_batch][0]
        batch_size = tar.shape[0]
        for p_data in range(batch_size):
            idx = ids[p_data]
            if idx == 3:
                # get output image
                out_img = res[p_data]
                out_img = np.transpose(out_img, (1, 2, 0))
                out_img = out_img * 255
                out_img = out_img.astype('uint8')
                # get target image
                tar_out_img = tar[p_data]
                tar_out_img = np.transpose(tar_out_img, (1, 2, 0))
                tar_out_img = tar_out_img * 255
                tar_out_img = tar_out_img.astype('uint8')
                # tar_out_img = cv2.cvtColor(tar_out_img, cv2.COLOR_RGB2BGR)
                # get data image
                data_out_img = data[p_data]
                data_out_img = np.transpose(data_out_img, (1, 2, 0))
                data_out_img = data_out_img * 255
                data_out_img = data_out_img.astype('uint8')


                data_out_img = cv2.cvtColor(data_out_img, cv2.COLOR_RGB2BGR)
                blurred = cv2.GaussianBlur(data_out_img, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                # find out the green ball position on the input image
                greenmask = cv2.inRange(hsv, greenLower, greenUpper)
                greenmask = cv2.erode(greenmask, None, iterations=2)
                greenmask = cv2.dilate(greenmask, None, iterations=2)

                green_cnts = cv2.findContours(greenmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                green_cnts = imutils.grab_contours(green_cnts)
                if len(green_cnts) == 0:
                    index = index + 1
                    continue
                c = max(green_cnts, key=cv2.contourArea)
                ((x, y), g_input_radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                green_input_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # find the robot position on the input image
                blackmask = cv2.inRange(hsv, blackLower, blackUpper)
                blackmask = cv2.erode(blackmask, None, iterations=2)
                blackmask = cv2.dilate(blackmask, None, iterations=2)

                black_cnts = cv2.findContours(blackmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                black_cnts = imutils.grab_contours(black_cnts)
                if len(black_cnts) == 0:
                    index = index + 1
                    continue
                c = max(black_cnts, key=cv2.contourArea)
                ((x, y), r_input_radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                r_input_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # find the position of the robot (black ball) in the target image
                tar_out_img = cv2.cvtColor(tar_out_img, cv2.COLOR_RGB2BGR)
                blurred = cv2.GaussianBlur(tar_out_img, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                blackmask = cv2.inRange(hsv, blackLower, blackUpper)
                blackmask = cv2.erode(blackmask, None, iterations=2)
                blackmask = cv2.dilate(blackmask, None, iterations=2)

                black_cnts = cv2.findContours(blackmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                black_cnts = imutils.grab_contours(black_cnts)

                c = max(black_cnts, key=cv2.contourArea)
                ((x, y), r_tar_radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                r_tar_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


                # find the position of the robot (black ball) in the predicted image
                out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                blurred = cv2.GaussianBlur(out_img, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

                blackmask = cv2.inRange(hsv, blackLower, blackUpper)
                blackmask = cv2.erode(blackmask, None, iterations=2)
                blackmask = cv2.dilate(blackmask, None, iterations=2)

                black_cnts = cv2.findContours(blackmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                black_cnts = imutils.grab_contours(black_cnts)
                if len(black_cnts) == 0:
                    result.append(0)
                    print('False here')
                    index = index + 1
                    continue
                c = max(black_cnts, key=cv2.contourArea)
                ((x, y), r_pred_radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                r_pred_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                r_dis_pred_input = np.linalg.norm(np.array(r_pred_center) - np.array(r_input_center))

                if r_dis_pred_input > 0.5 * r_input_radius:
                    # find the largest / 2nd largest black contour on the image
                    black_cnts = cv2.findContours(blackmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    black_cnts = imutils.grab_contours(black_cnts)

                    area_array = []
                    for i, c in enumerate(black_cnts):
                        area = cv2.contourArea(c)
                        area_array.append(area)
                    #first sort the array by area
                    sorteddata = sorted(zip(area_array, black_cnts), key=lambda x: x[0], reverse=True)
                    #find the nth largest contour [n-1][1], in this case 2
                    pred_cnts = []
                    for i in range(len(sorteddata)):
                        if i < top_k_cnts:
                            pred_cnts.append(sorteddata[i][1])
                            cv2.drawContours(out_img, sorteddata[i][1], -1, (255, 0, 0), 2)
                        else:
                            break

                    min_dis = 1000
                    for p_c in pred_cnts:
                        for p_p in p_c:
                            dis = np.linalg.norm(np.array(p_p) - np.array(green_input_center))
                            if dis < min_dis:
                                min_dis = dis
                                min_point = p_p
                    cv2.circle(out_img, (min_point[0][0], min_point[0][1]), 2, (0, 0, 255), -1)
                    if min_dis <= r_input_radius * 2:
                        # print('True')
                        result.append(1)
                    else:
                        print('False')
                        result.append(0)


                else:
                    greenmask = cv2.inRange(hsv, greenLower, greenUpper)
                    greenmask = cv2.erode(greenmask, None, iterations=2)
                    greenmask = cv2.dilate(greenmask, None, iterations=2)
                    green_cnts = cv2.findContours(greenmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    green_cnts = imutils.grab_contours(green_cnts)
                    if len(green_cnts) == 0:
                        index = index + 1
                        continue
                    c = max(green_cnts, key=cv2.contourArea)
                    ((x, y), g_pred_radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    green_pred_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    g_dis_pred_tar = np.linalg.norm(np.array(green_input_center) - np.array(green_pred_center))
                    if g_dis_pred_tar <= g_input_radius * 0.5:
                        result.append(1)
                    else:
                        print('False')
                        result.append(0)

                numpy_vertical_concat = np.concatenate((data_out_img, tar_out_img, out_img), axis=1)
                if result[-1] == 1:
                    cv2.imwrite(os.path.join(example_measure_imgs_path, str(index) + '.png'), numpy_vertical_concat)

                index = index + 1
                # cv2.imshow('frame', numpy_vertical_concat)
                # cv2.waitKey()

accuracy = sum(result) / len(result) * 100
print('total: ', len(result))
print(accuracy)