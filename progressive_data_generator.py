
import os
import cv2
import time
import shutil
import imutils
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from imutils.video import VideoStream

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points

redLower = (0, 106, 46)
redUpper = (46, 255, 255)

raw_base_path = 'raw_data'
data_base_path = 'data_imgs_progressive'
array_base_path = 'data_arr_progressive'
# otherwise, grab a reference to the video file



def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

mkdir(data_base_path)
mkdir(array_base_path)



# get the data_idx that are used for testing
test_ids = np.load('test_ids.npy')
test_ids = list(test_ids)

filelist = os.listdir(raw_base_path)


data_idx = 0
for p_file in tqdm(filelist):
    if data_idx not in test_ids:
        data_idx = data_idx + 1
        continue
    all_imgs = []
    vs = cv2.VideoCapture(os.path.join(raw_base_path, p_file))
    # allow the camera or video file to warm up
    time.sleep(2.0)
    # keep looping
    idx = 0
    cur_path_length = 0
    half_total_path_length = 0
    while True:
        if idx <= 9:
            idx = idx + 1
            continue
        # grab the current frame
        frame = vs.read()

        # handle the frame from VideoCapture or VideoStream
        frame = frame[1]
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        redmask = cv2.inRange(hsv, redLower, redUpper)
        redmask = cv2.erode(redmask, None, iterations=2)
        redmask = cv2.dilate(redmask, None, iterations=2)
        red_cnts = cv2.findContours(redmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_cnts = imutils.grab_contours(red_cnts)

        if len(red_cnts) == 0:
            continue


        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        # cv2.rectangle(frame, (0, 27), (600, 450-15), (0, 0, 255), 2)
        frame = frame[27:(450-15), 0:600]

        w, h, c = frame.shape

        if frame is not None:
            all_imgs.append(frame.reshape(1, -1))

    # show the frame to our screen
    # cur_frame = np.array(all_imgs.copy()[:int(len(all_imgs) / 4)]).squeeze(1)
    # cur_frame = np.amin(cur_frame, axis=0)


    mkdir(os.path.join(data_base_path, str(data_idx)))
    mkdir(os.path.join(array_base_path, str(data_idx)))
    for i in range(len(all_imgs)-1):
        frame = np.array(all_imgs[:i+1]).squeeze(1)
        frame = np.amin(frame, axis=0)
        frame = frame.reshape(w, h, c)
        frame = cv2.resize(frame, (64, 64), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(data_base_path, str(data_idx), 'frame_' + str(i) + '.png'), frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        np.save(os.path.join(array_base_path, str(data_idx), 'frame_' + str(i) + '.npy'), frame)

    data_idx = data_idx + 1
    vs.release()
    cv2.destroyAllWindows()


# ffmpeg -i top_366_0.avi -vf scale=256:256 tmp_imgs/_%04d.jpg -loglevel quiet