import cv2

import numpy as np
import time


if __name__ == "__main__":
    try:
        #load video and point 3d
        cap = cv2.VideoCapture('/home/bao/Desktop/video/source_train/trai.avi')
        points_3d = np.load("/home/bao/Desktop/video/source_train/trai.npy")

        #setup
        name_video_p3d = 'sp/data_im'
        # x,y = 275,190
        # w,h = 30,30
        x,y = 322,123
        w,h = 30,17

        out = cv2.VideoWriter(name_video_p3d +'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (848,480))
        save = False
        cnt = 0

        element = np.arange(15).reshape((15,1)).tolist()
        element2 = np.arange(15).reshape((15,1)).tolist()
 
        element_th = 0
        matrix = []
        point_n = [ 0 ]
        #read frame from video
        for index in range(int(cap.get(7))):
                ret,frame = cap.read()

                i = "{}".format(index)
                cv2.putText(frame,'frame: '+str(i) + ' SAVING',(30,45),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1,)
                cv2.imshow('frame red:',frame)
                cnt+=1
                out.write(frame)
                element[cnt-1] = [points_3d[int(index)][0], points_3d[int(index)][1],points_3d[int(index)][2]]
                if cnt >= 15:
                    point_n = element[0]
                    for j in range(len(element)):
                        element2[j] = [element[j][0] - point_n[0], element[j][1] - point_n[1],
                                                                    element[j][2] - point_n[2]]
                    cnt = 0
                    matrix.append(element2)
                print('saving')
                # time.sleep(0.25)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

    finally:
        matrix= np.array(matrix)
        np.save(name_video_p3d,matrix)
        print(matrix)
        cv2.destroyAllWindows()