import cv2
import os
import time
from pathlib import Path


def get_frames(filename):

    vid = cv2.VideoCapture(filename)
    #frame_rate = 10
    i = 0
    
    vid.open(filename)
    start_time = time.time()
    fps = vid.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)

    while vid.isOpened():
        i+=1
        ret, frame = vid.read()
        if not ret:
            break

        #if time_elapsed > 1./frame_rate:
            # print(time_elapsed)
            #prev = time.time()
            #cv2.imwrite('./data/sample1/test_'+str(i)+'.jpg', frame)
            #i += 1
        cv2.imwrite('./'+str(i)+'.jpg', frame)
        

    time_elapsed = time.time() - start_time
"""     
    print("time elapsed:", time_elapsed)
    print("frames:", i)
    print("fps:", i/time_elapsed)
 """
    #vid.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    print(cv2.__version__)    
    get_frames('./something_videos/2.webm')