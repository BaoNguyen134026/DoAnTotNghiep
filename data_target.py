import numpy as np
import cv2
from random import randint
#create data and target
data = []
target = []
f1 = np.load("/home/bao/Desktop/video/sp/sp_trai.npy")
f2 = np.load("/home/bao/Desktop/video/sp/sp_phai.npy")
f3 = np.load("/home/bao/Desktop/video/sp/sp_xuong.npy")
print(len(f1))
print(len(f2))
print(len(f3))

for i in range(1,len(f1)):
    data.append(f1[i])
    target.append(1)

for i in range(1,len(f2)):
    data.append(f2[i])
    target.append(2)

for i in range(1,len(f3)):
    data.append(f3[i])
    target.append(3)

data = np.array(data)
target = np.array(target)

np.save('/home/bao/Desktop/video/sp/data',data)
np.save('/home/bao/Desktop/video/sp/target',target)
print(len(data))
print(target)


