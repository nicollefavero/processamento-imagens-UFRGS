import numpy as np
from cv2 import cv2

def cap_and_show():
    camera = 0
    cap = cv2.VideoCapture(camera)

    if(cap.open(camera)):
        frame = np.array(cap.grab(), dtype=np.uint8)

        if(frame.size == 0):
            print("BREAK")
        else:
            cv2.imwrite("image_output.jpeg", frame)
    
    cap.release()

# cap_and_show()
camera = 1
cap = cv2.VideoCapture(camera)
print(cap.read())
# cv2.imwrite("img.jpg", img)