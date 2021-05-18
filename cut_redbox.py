import cv2

import numpy as np
import time

def detect_redbox(frame,x,y,w,h):
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    red_chanel = frame[:,:,2]

    # cv2.rectangle(gray,p1,p2,(0,0,255),1)

    #Crop box red
    crop = red_chanel[y:y+h,x:x+w] 
    crop = cv2.GaussianBlur(crop,(5,5),0)

    _,thresh = cv2.threshold(crop,150,255,cv2.THRESH_BINARY)

    # print(np.count_nonzero(thresh==255))
    if np.count_nonzero(thresh==255)>300:# >1 is okey
        return True
    else: return False

if __name__ == "__main__":
    try:
        #load video and point 3d
        cap = cv2.VideoCapture('/home/bao/Desktop/DOAN/doan_totnghiep/dongtac_xoaytron.avi')
        points_3d = np.load("dongtac_xoaytron.npy")

        #setup
        name_video_p3d = 'sp/dongtacxoaytron'
        # x,y = 275,190
        # w,h = 30,30
        x,y = 322,123
        w,h = 30,17

        out = cv2.VideoWriter(name_video_p3d +'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (848,480))
        save = False
        cnt = 0

        element = np.arange(15).reshape((15,1)).tolist()
               
        element_th = 0
        matrix = []
        
        #read frame from video
        for index in range(int(cap.get(7))):
                index = index + 5
                ret,frame = cap.read()
                if detect_redbox(frame,x,y,w,h) == True:
                    print(index)
                    if cnt < 1:
                        save = True
                        point_n = points_3d[int(index)]
                if save == True:
                    i = "{}".format(index)
                    cv2.putText(frame,'frame: '+str(i) + ' SAVING',(30,45),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1,)
                    cv2.imshow('frame red:',frame)
                    cnt+=1
                    out.write(frame)
                    element[cnt-1] = [points_3d[int(index)][0] - point_n[0], points_3d[int(index)][1] - point_n[1],points_3d[int(index)][2] - point_n[2]]
                    # print(element)
                    if cnt >= 15:
                        save = False
                        cnt = 0
                        matrix.append(element)
                else:
                    i = "{}".format(index)
                    cv2.putText(frame,'frame: '+str(i),(30,45),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1,)
                    cv2.imshow('frame red:',frame)
                # time.sleep(0.25)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

    finally:
        matrix= np.array(matrix)
        np.save(name_video_p3d,matrix)
        print(matrix)
        cv2.destroyAllWindows()