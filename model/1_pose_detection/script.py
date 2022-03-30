import time

import cv2
import numpy as np
import requests

camera = cv2.VideoCapture(0)  # use 0 for web camera
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    # Capture frame-by-frame
    # success, frame = camera.read()  # read the camera frame
    success, frame = True, cv2.imread('test/20220312_120226.jpg')
    if not success:
        break
    else:
        # encode (This Part in the script that calls the api frame by frame)
        # scale_percent = 40  # percent of original size
        # width = int(frame.shape[1] * scale_percent / 100)
        # height = int(frame.shape[0] * scale_percent / 100)
        # dim = (width, height)

        # resize image
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        data_encode = np.array(frame).tolist()
        data = {"data": data_encode}
        # sending post request and saving response as response object
        start = time.time()
        r = requests.post(url='http://127.0.0.1:5000/pose', json=data)
        end = time.time()
        print(end - start, 's')
        nparr = np.array(r.json()["image"], np.uint8)
        # cv2.imshow('im', nparr)
        cv2.imwrite('result/20220312_120226.png', nparr )
        break
        # cv2.imshow('im', cv2.resize(nparr,(1920,1080)))
        # cv2.waitKey(1)
