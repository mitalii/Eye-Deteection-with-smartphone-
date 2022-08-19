import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import dlib


def gamma_correct(img, gamma=0.6):
    img = img/255.
    img = img**gamma
    img = (img*255).astype(np.uint8)
    return img
    
def detect_eyes(predictor, gray, rect):
    shape = eyes_detector(gray, rect)

    pts_l = []
    pts_r = []
    for ii in range(36,42):
        pts_l.append([shape.part(ii).x, shape.part(ii).y])
    for ii in range(42,48):
        pts_r.append([shape.part(ii).x, shape.part(ii).y])
        
    pts_left = np.array(pts_l)
    pts_right = np.array(pts_r)
    
    return pts_left, pts_right

def locate_pupil(gray, pts):
    x1 = np.amin(pts[:,0])
    y1 = np.amin(pts[:,1])
    x2 = np.amax(pts[:,0])
    y2 = np.amax(pts[:,1])

    eye_img = gray[y1:y2, x1:x2]
    mask = np.ones(eye_img.shape, dtype=eye_img.dtype)*255
    for ii in range(6):
        pts[ii,0] -= x1
        pts[ii,1] -= y1

    cv2.drawContours(mask, [pts], 0, 0, thickness=cv2.FILLED)
    
    
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, se)

    eye_img = eye_img * ((255-mask)//255)
    w,h = eye_img.shape[1], eye_img.shape[0]
    eye_img[:, 0:w//5] = 0
    eye_img[:, 4*w//5:] = 0
    
    mx = np.amax(eye_img)
    res = (eye_img < 0.12*mx).astype(np.uint8)
    res[:, 0:w//5] = 0
    res[:, 4*w//5:] = 0
    res = res * ((255-mask)//255)
    '''
    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.show()
    '''
    #se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    #res = cv2.morphologyEx(res, cv2.MORPH_OPEN, se)
    
    contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    
    if c_x == 0 and c_y == 0:
        c_x = eye_img.shape[1]//2
        c_y = eye_img.shape[0]//2
        c_radius = 1
    if c_radius < 1:
        c_radius = 1
    
    return (int(c_x), int(c_y), int(c_radius)), (int(x1), int(y1))


try:

    cap = cv2.VideoCapture(0)

    count = 0
    left_area = []
    right_area = []
    left_radius = []
    right_radius = []

    face_detector = dlib.get_frontal_face_detector()
    eyes_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while cap != None and cap.isOpened() and count < 100:

        ret, frame = cap.read()
        img = frame[:,::-1,:].copy()
        
        
        count += 1
        if ret: # and count%25 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray, 1)    
            if len(faces) > 0:
                cv2.rectangle(gray, (faces[0].left(), faces[0].top()), (faces[0].right(), faces[0].bottom()), (0,255,0), 2)
                
                pts_l, pts_r = detect_eyes(eyes_detector, gray, faces[0])

                cir_l, (xl1, yl1) = locate_pupil(gray, pts_l.copy())
                cir_r, (xr1, yr1) = locate_pupil(gray, pts_r.copy())
                
                cv2.circle(img, (cir_l[0]+xl1, cir_l[1]+yl1), cir_l[-1], (255,255,0), 2)
                cv2.circle(img, (cir_r[0]+xr1, cir_r[1]+yr1), cir_r[-1], (255,255,0), 2)
                
                left_area.append(3.1416*(cir_l[2]**2))
                right_area.append(3.1416*(cir_r[2]**2))
        
            cv2.imshow('Try', img)
        cv2.waitKey(1) #time.sleep(0.1)
            
        
    cv2.destroyAllWindows()

    fig = plt.figure()
    plt.plot(left_area,'r')
    plt.plot(right_area,'b')
    plt.xlabel('time (seconds)')
    plt.ylabel('pupil area (no. of pixels)')
    plt.show()

except Exception as ex:
    print(ex)