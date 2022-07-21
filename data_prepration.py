from cgitb import handler
import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from prometheus_client import Counter

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 2)

offset = 20
img_size = 300
counter = 0
# This is the path where all the photos of alphabates are stored in seperate folder like A, B, C, ........., Z
# I have added on;ly four you can add as many as you want oviously till z
folder = "D:\Python\Project\Deep Learning\Computer vision\Sign Language\A"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # Croped imgae from raw camera video
        img_crop = img[y-offset:y + h+offset, x-offset:x + w+offset]
        cv2.imshow("Image croped", img_crop)

        # This is the section which we will put the image to get it in perfect square
        img_white = np.ones((img_size, img_size, 3), np.uint8) * 255


        # After putting the image on top of white section
        aspect_ratio = h/w

        if aspect_ratio > 1:
            k = img_size/h
            w_calculated = math.ceil(k*w)
            img_resize = cv2.resize(img_crop, (w_calculated, img_size))
            img_resize_shape = img_resize.shape
            w_gap = math.ceil((img_size - w_calculated) / 2)
            img_white[:, w_gap:w_calculated + w_gap] = img_resize

        else:
            k = img_size/w
            h_calculated = math.ceil(k*h)
            img_resize = cv2.resize(img_crop, (img_size, h_calculated))
            img_resize_shape = img_resize.shape
            h_gap = math.ceil((img_size - h_calculated) / 2)
            img_white[h_gap:h_calculated + h_gap, :] = img_resize
        
        cv2.imshow("Image white", img_white)


    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):     
        counter += 1
        cv2.imwrite(f"{folder}/image_{time.time()}.jpg", img_white)
        print(counter)








 
