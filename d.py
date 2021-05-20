import math as m
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics

import numpy as np
import pickle
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime as dt

def draw(d):
    global first_loop, cnt, d15s, d3s
    if first_loop == True:
        #  original 15 point3d
        d15s[cnt - 1] = d
        cnt+=1
        if cnt >= 15:
            first_loop = False
            cnt = 0
            y = d15s
            x = range(15)
            plt.plot(x, y, color = 'black' , linewidth=2,
            label="dongtac")
            plt.show()
            plt.close()

    else:
        # print('test ne')
        # print('hieu x = \n',point_3d[0]-fifteen_temporary_points[14][0])
        # print('hieu y = \n',point_3d[1]-fifteen_temporary_points[14][1])
        # print('hieu z = \n',point_3d[2]-fifteen_temporary_points[14][2])
        
        if cnt <= 2:
                          
            d3s[cnt] = d
            cnt+=1

        else:
            for ii in range(0,12):
                d15s[int(ii)] = d
                
            for ii in range(12,15):
                d15s[int(ii)] = d
            y = d15s
            x = range(15)
            plt.plot(x, y, color = 'black' , linewidth=2,
            label="dongtac")
            plt.show()
            plt.close()
def animate(d, xs, ys):

    # Read temperature (Celsius) from TMP102
    
    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%M:%S'))
    ys.append(d)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('D')
    plt.ylabel('D')
if __name__ == "__main__":
    try:
        cap1 = cv2.VideoCapture('/home/bao/Desktop/DATN/videosource/dongtac_xoaytron.avi')

        f1 = np.load("/home/bao/Desktop/DATN/npy_source/dongtac_xoaytron.npy")
        loaded_model = pickle.load(open('/home/bao/Desktop/DATN/sp/traindongtac.sav', 'rb'))

        first_loop = True
        d3s = np.arange(3).reshape((3,1)).tolist()
        d15s = np.arange(15).reshape((15,1)).tolist()
        cnt = 0
        point_3di = np.arange(3).reshape((3,))
        y = []
        a = 0
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        xs = []
        ys = []
        for index in range(int(cap1.get(7))):

            ret,frame = cap1.read()
            point_3d = f1[int(index)]
            if point_3d is not None:
                # print('p3d = ',point_3d)
                # print('p3di = ',point_3di)
                d = m.sqrt(m.pow((point_3d[0]-point_3di[0]),2)) +   m.pow((point_3d[1]-point_3di[1]),2)+m.pow((point_3d[2]-point_3di[2]),2)               
                point_3di = point_3d
                print(d)
                animate(d,xs,ys)
                # y.append(d)
                # a+=1
                # print('y= ',y)
                # print('d=',d)
                pass
            cv2.imshow('video',frame)

            ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000)
            plt.show()

            if cv2.waitKey(30) ==27 :
                plt.show()

                break
        # x = range(a)
        # plt.plot(x, y, color = 'black' , linewidth=2,
        #     label="dongtac")
        # plt.show()
        # plt.close()
    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))