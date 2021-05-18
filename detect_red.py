import cv2

import numpy as np

#load video and point 3d
cap = cv2.VideoCapture('/opt/cubemos/skeleton_tracking/samples/python/outpy2.avi')
points_3d = np.load("save3d.npy")

#size video
print('frame video:',cap.get(7))

#tao bien

xx = False
count_frame=0
element = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
element_th = 0
# x,y = 120,203
# w,h = 60,27
x,y = 525,335
w,h = 30,30

matrix = []
def detect_redbox(frame,x,y,w,h):
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    red_chanel = frame[:,:,2]
    # cv2.rectangle(gray,p1,p2,(0,0,255),1)

    #Crop box red

    crop = red_chanel[y:y+h,x:x+w] 
    crop = cv2.GaussianBlur(crop,(5,5),0)
    _,thresh = cv2.threshold(crop,150,255,cv2.THRESH_BINARY)

    print('count',np.count_nonzero(thresh>200))

    if np.count_nonzero(thresh>200)>230:# >1 is okey
        return True
    else: return False


for index in range(int(cap.get(7))):

    _,frame=cap.read()
    if int(index) == 419:

        cv2.imshow('thresh',detect_redbox(frame,120,203,60,27))
        cv2.rectangle(frame,(525,335),(555,365),(0,0,255),1)
        cv2.imshow('show',frame)
        print(detect_redbox(frame,120,203,60,27))
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    # for index in range(int(cap.get(7))):
        # result = frame.copy()
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lower = np.array([155,25,0])
        # upper = np.array([179,255,255])
        # mask = cv2.inRange(image, lower, upper)
        # result = cv2.bitwise_and(result, result, mask=mask)

        # cv2.imshow('frame', frame[:,:,2])

        # cv2.imshow('mask', mask)
        # cv2.imshow('result', result)
        # cv2.waitKey()





# Closes all the frames
cv2.destroyAllWindows()
