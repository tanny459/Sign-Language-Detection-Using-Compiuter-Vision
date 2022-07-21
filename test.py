import cv2
import math
from cv2 import putText
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
from prometheus_client import Counter

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 2)
Classifier = Classifier("D:\Python\Project\Deep Learning\Computer vision\Sign Language\Model\keras_model.h5",
                        "D:\Python\Project\Deep Learning\Computer vision\Sign Language\Model\labels.txt")

offset = 20
img_size = 300
counter = 0
# You cand add all the alphabates if you wish right down below in the list of labels
labels = ["A", "B", "C", "D"]

while True:
    success, img = cap.read()
    image_output = img.copy()
    hands = detector.findHands(img, draw = False)
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
            predection, index = Classifier.getPrediction(img_white, draw = False)
            print(predection, index)
            

        else:
            k = img_size/w
            h_calculated = math.ceil(k*h)
            img_resize = cv2.resize(img_crop, (img_size, h_calculated))
            img_resize_shape = img_resize.shape
            h_gap = math.ceil((img_size - h_calculated) / 2)
            img_white[h_gap:h_calculated + h_gap, :] = img_resize
            predection, index = Classifier.getPrediction(img_white, draw = False)

        cv2.imshow("Image white", img_white)

        cv2.rectangle(image_output, (x - offset, y - offset - 50), 
                    (x - offset + 80, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(image_output, labels[index], (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        1.7,(255, 255, 255), 2)
        cv2.rectangle(image_output, (x - offset, y - offset), 
                        (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Image", image_output)
    cv2.waitKey(1)

