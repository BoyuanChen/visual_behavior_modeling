
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
greenLower = (41, 54, 110)
greenUpper = (74, 255, 255)
blackLower = (0, 0, 0)
blackUpper = (255, 255, 88)
raw_base_path = 'raw_data'
data_base_path = 'data_imgs'
array_base_path = 'data_arr'
# otherwise, grab a reference to the video file



def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

mkdir(data_base_path)
mkdir(array_base_path)



filelist = os.listdir(raw_base_path)


data_idx = 0
for p_file in tqdm(filelist):
    pts = []
    pts_frame = OrderedDict()
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

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        # cv2.rectangle(frame, (0, 27), (600, 450-15), (0, 0, 255), 2)
        frame = frame[27:(450-15), 0:600]
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # green
        greenmask = cv2.inRange(hsv, greenLower, greenUpper)
        greenmask = cv2.erode(greenmask, None, iterations=2)
        greenmask = cv2.dilate(greenmask, None, iterations=2)
        # black
        blackmask = cv2.inRange(hsv, blackLower, blackUpper)
        blackmask = cv2.erode(blackmask, None, iterations=2)
        blackmask = cv2.dilate(blackmask, None, iterations=2)
        # red
        redmask = cv2.inRange(hsv, redLower, redUpper)
        redmask = cv2.erode(redmask, None, iterations=2)
        redmask = cv2.dilate(redmask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        # green
        green_cnts = cv2.findContours(greenmask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        green_cnts = imutils.grab_contours(green_cnts)
        # black
        black_cnts = cv2.findContours(blackmask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        black_cnts = imutils.grab_contours(black_cnts)
        # red
        red_cnts = cv2.findContours(redmask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        red_cnts = imutils.grab_contours(red_cnts)

        green_center = None
        black_center = None
        red_center = None

        # only proceed if at least one contour was found
        if len(green_cnts) > 0 and len(red_cnts) > 0 and len(black_cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            # green
            c = max(green_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # # only proceed if the radius meets a minimum size
            # if radius > 10:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     cv2.circle(frame, (int(x), int(y)), int(radius),
            #         (0, 255, 0), 2)
            #     cv2.circle(frame, green_center, 5, (0, 255, 0), -1)
            # black
            c = max(black_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            black_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # # only proceed if the radius meets a minimum size
            # if radius > 10:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     cv2.circle(frame, (int(x), int(y)), int(radius),
            #         (0, 0, 0), 2)
            #     cv2.circle(frame, black_center, 5, (0, 0, 0), -1)
            # red
            c = max(red_cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            red_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            # if radius > 10:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     cv2.circle(frame, (int(x), int(y)), int(radius),
            #         (0, 0, 255), 2)
            #     cv2.circle(frame, red_center, 5, (0, 0, 255), -1)

            # dis_robot_green = np.linalg.norm(np.array(black_center) - np.array(green_center))
            # if dis_robot_green < 86:
            #     assert False
            # print(dis_robot_green)
            if green_center is not None:
                target_center = green_center

        # update the points queue
        pts.append(black_center)

        for i in range(len(pts)):
            if pts[i] is not None:
                init_robot_pos = pts[i]
                break
        if black_center is not None:
            half_total_path_length = np.linalg.norm(np.array(init_robot_pos) - np.array(target_center)) * 0.5
            cur_path_length = np.linalg.norm(np.array(init_robot_pos) - np.array(black_center))
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            # thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 0), 5)
        # if cur_path_length != 0 and half_total_path_length != 0 and frame is not None:
        #     if cur_path_length <= half_total_path_length + 10 and cur_path_length >= half_total_path_length - 10:
        #         cur_frame = frame.copy()
        pts_frame[black_center] = frame.copy()
        if frame is not None:
            end_frame = frame.copy()

    # show the frame to our screen
    # num_frames = len(pts_frame.keys())
    middle_pts = list(pts_frame.keys())[0]
    cur_frame = pts_frame[middle_pts]
    end_frame = cv2.resize(end_frame, (64, 64), interpolation = cv2.INTER_AREA)
    cur_frame = cv2.resize(cur_frame, (64, 64), interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(data_base_path, 'end_' + str(data_idx) + '.png'), end_frame)
    cv2.imwrite(os.path.join(data_base_path, 'cur_' + str(data_idx) + '.png'), cur_frame)
    end_frame = cv2.cvtColor(end_frame, cv2.COLOR_BGR2RGB)
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
    np.save(os.path.join(array_base_path, 'end_' + str(data_idx) + '.npy'), end_frame)
    np.save(os.path.join(array_base_path, 'cur_' + str(data_idx) + '.npy'), cur_frame)
    data_idx = data_idx + 1

    vs.release()
    cv2.destroyAllWindows()