import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import dlib
import datetime
import os


def gamma_correct(img, gamma=0.6):
    img = img / 255.
    img = img ** gamma
    img = (img * 255).astype(np.uint8)
    return img


def detect_eyes(predictor, gray, rect):
    shape = eyes_detector(gray, rect)

    pts_l = []
    pts_r = []
    for ii in range(36, 42):
        pts_l.append([shape.part(ii).x, shape.part(ii).y])
    for ii in range(42, 48):
        pts_r.append([shape.part(ii).x, shape.part(ii).y])

    pts_left = np.array(pts_l)
    pts_right = np.array(pts_r)

    return pts_left, pts_right


def locate_pupil(gray, pts):
    x1 = np.amin(pts[:, 0])
    y1 = np.amin(pts[:, 1])
    x2 = np.amax(pts[:, 0])
    y2 = np.amax(pts[:, 1])

    eye_img = gray[y1:y2, x1:x2]
    mask = np.ones(eye_img.shape, dtype=eye_img.dtype) * 255
    for ii in range(6):
        pts[ii, 0] -= x1
        pts[ii, 1] -= y1

    cv2.drawContours(mask, [pts], 0, 0, thickness=cv2.FILLED)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, se)

    # detect iris
    eye_img = eye_img * ((255 - mask) // 255)
    w, h = eye_img.shape[1], eye_img.shape[0]
    # eye_img[:, 0:w//5] = 0
    # eye_img[:, 4*w//5:] = 0

    mx = np.amax(eye_img)
    res = (eye_img < 0.5 * mx).astype(np.uint8)
    res = res * ((255 - mask) // 255)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, se)
    contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.zeros(mask.shape, dtype=mask.dtype)

    # retain largest region to segment iris
    c_area = 0
    c_x = 0
    c_y = 0
    c_radius = 1
    for cntr in contours:
        area = cv2.contourArea(cntr)
        (cx, cy), radius = cv2.minEnclosingCircle(cntr)
        if area >= c_area:
            c_x = int(cx)
            c_y = int(cy)
            c_area = area
            c_radius = int(radius)

    if c_radius < 1:
        c_radius = 3
    cv2.circle(mask2, (c_x, c_y), c_radius, 255, -1)
    # mask2 = (mask//255)*(mask2//255)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, se)
    idx = np.argwhere(mask2)
    min_x = np.amin(idx[:, 1])
    max_x = np.amax(idx[:, 1])
    min_y = np.amin(idx[:, 0])
    max_y = np.amax(idx[:, 0])

    avg_dist = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2 + 0.1)

    # now using mask2, segment iris and find threshold in this region
    iris_img = eye_img * (mask2 // 255)
    pupil = iris_img < (iris_img[c_y, c_x] + 4)
    pupil = pupil * (mask2 // 255)

    # locate pupil in pupil image
    idxs = np.argwhere(mask2)
    dist = 0
    for idx in idxs:
        tmp = np.sqrt((c_x - idx[1]) ** 2 + (c_y - idx[0]) ** 2 + 0.1)
        if tmp > dist:
            dist = tmp

    if dist > avg_dist / 4:
        c_radius = int(avg_dist / 4)
    else:
        c_radius = int(dist)

    '''
    plt.figure()
    plt.imshow(pupil, cmap='gray')
    plt.show()
    '''

    '''
    contours, _ = cv2.findContours(pupil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c_area = 0
    c_x, c_y, c_radius = 0,0,1
    c_dist = 10000

    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        area = cv2.contourArea(cntr)
        (cx, cy), radius = cv2.minEnclosingCircle(cntr)
        dist = abs((y+h/2) - eye_img.shape[0]/2)
        if area >= c_area and dist < c_dist:
            c_radius = radius
            c_x = cx
            c_y = cy
            c_dist = dist
            c_area = area
    '''
    if c_x == 0 and c_y == 0:
        c_x = eye_img.shape[1] // 2
        c_y = eye_img.shape[0] // 2
        c_radius = 1
    if c_radius < 1:
        c_radius = 1

    return (int(c_x), int(c_y), int(c_radius)), (int(x1), int(y1))


#############################################################
########### MAIN --------------------------------------------


try:
    video_path = 'participants videos/sahar_new.mp4'
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()


    count = 0
    left_area = []
    right_area = []
    left_radius = []
    right_radius = []

    face_detector = dlib.get_frontal_face_detector()
    eyes_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    writer = cv2.VideoWriter('./' + video_path[:-4] + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24,
                             (frame.shape[1], frame.shape[0]))

    while cap != None and cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        img = frame.copy()   #[::-1,:,:].copy()
        # agar image ulta ho jaey to uncomment this
       # img = frame[::-1, :, :].copy()

        # ----------------------------------------------------------
        # comment these 3 lines below if you want to run whole video
        #if count > 1000:  # this is just to run the video for short amount of time
            # after processing 500 frames it will quit
          #  break
        # ----------------------------------------------------------
        count += 1
        if count % 5 == 0:  # this is to process every 5th frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray, 1)
            if len(faces) > 0:
                cv2.rectangle(gray, (faces[0].left(), faces[0].top()), (faces[0].right(), faces[0].bottom()),
                              (0, 255, 0), 2)

                pts_l, pts_r = detect_eyes(eyes_detector, gray, faces[0])

                cir_l, (xl1, yl1) = locate_pupil(gray, pts_l.copy())
                cir_r, (xr1, yr1) = locate_pupil(gray, pts_r.copy())

                cv2.circle(img, (cir_l[0] + xl1, cir_l[1] + yl1), cir_l[-1], (255, 255, 0), 2)
                cv2.circle(img, (cir_r[0] + xr1, cir_r[1] + yr1), cir_r[-1], (255, 255, 0), 2)

                left_area.append(3.1416 * (cir_l[2] ** 2))
                right_area.append(3.1416 * (cir_r[2] ** 2))

                left_radius.append(cir_l[2])
                right_radius.append(cir_r[2])

            cv2.imshow('Try', img)
            writer.write(img)
        if cv2.waitKey(1) == ord('q'):
            break

    writer.release()
    cv2.destroyAllWindows()

    fid = open('./' + video_path[:-4] + '.csv', 'w')
    fid.write('Frame_No,left_pupil_area,right_pupil_area,left_pupil_radius,right_pupil_radius\n')
    for ii in range(len(left_radius)):
        fid.write(
            str(ii + 1) + ',' + str(left_area[ii]) + ',' + str(right_area[ii]) + ',' + str(left_radius[ii]) + ',' + str(
                right_radius[ii]) + '\n')
    fid.close()

    fig = plt.figure()
    plt.plot(left_area, 'r')
    plt.plot(right_area, 'b')
    plt.xlabel('time (seconds)')
    plt.ylabel('pupil area (no. of pixels)')
    plt.show()

except Exception as ex:
    print(ex)