import time

import cv2
import numpy as np
from PIL import ImageGrab


def process_img(image):
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # edge detection
    # processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img


def grab_screen():
    screen = np.array(ImageGrab.grab(bbox=[0, 0, 800, 600]))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    return screen


last_time = time.time()
while True:
    screen = grab_screen()
    print('{} FPS'.format(1 / (time.time() - last_time)))
    last_time = time.time()
    cv2.imshow('window', screen)
    # cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
