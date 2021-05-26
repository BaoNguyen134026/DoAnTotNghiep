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

    _,thresh = cv2.threshold(crop,200,255,cv2.THRESH_BINARY)

    # print(np.count_nonzero(thresh==255))
    if np.count_nonzero(thresh==255)>500:# >1 is okey
        return True
    else: return False


if __name__ == "__main__":
    try:
        #load video and point 3d
        cap = cv2.VideoCapture('/home/bao/Desktop/video/source_train/xuong.avi')
        points_3d = np.load("/home/bao/Desktop/video/source_train/xuong.npy")
        #setup
        name_video_p3d = '/home/bao/Desktop/video/sp/sp_xuong'
        # print(points_3d)
        x,y,w,h = 550,228,30,20
        out = cv2.VideoWriter(name_video_p3d +'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (848,480))
        save = False
        cnt = 0
        element = np.arange(15).reshape((15,))
        # print(element)
        element_th = 0
        matrix = np.arange(100*15*3).reshape((100,15,3)).tolist()
        k = 0
        #read frame from video
        for index in range(int(cap.get(7))-8):
                index = index + 8
                ret,frame = cap.read()
                if k == 100: break
                if detect_redbox(frame,x,y,w,h) == True:
                    if cnt < 1:
                        save = True
                        point_n = points_3d[int(index)]
                if save == True:
                    i = "{}".format(index)
                    cv2.putText(frame,'frame: '+str(i)+' /'+str(k)+'/ '+ ' SAVING '+str(cnt),(30,45),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1,)
                    cv2.imshow('frame red:',frame)
                    cnt+=1
                    out.write(frame)

                    matrix[k][cnt-1] = [points_3d[index][0] - point_n[0],points_3d[index][1] - point_n[1],points_3d[index][2] - point_n[2]]
                    
                    if cnt >= 15:

                        # matrix[k]= element
                        k+=1
                        print(k)
                        save = False
                        cnt = 0

                else:
                    i = "{}".format(index)
                    cv2.putText(frame,'frame: '+str(i),(30,45),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1,)
                    cv2.imshow('frame red:',frame)
                # time.sleep(0.4)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

    finally:
        matrix= np.array(matrix)
        np.save(name_video_p3d,matrix)
        print(matrix)
        cv2.destroyAllWindows()