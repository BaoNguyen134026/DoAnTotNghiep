import numpy as np
import cv2
from random import randint
#create data and target
data = []
target = []
f1 = np.load("sp/dongtacgattrai.npy")
f2 = np.load("sp/dongtacgatphai.npy")
f3 = np.load("sp/dongtacxoaytron.npy")
f4 = np.load("sp/data_im.npy")
# print(len(f1))
# print(len(f2))
# print(len(f3))
# print(len(f4))

for i in range(10,len(f1)):
    data.append(f1[i])
    target.append(1)

for i in range(10,len(f2)):
    data.append(f2[i])
    target.append(2)

for i in range(15,len(f3)):
    data.append(f3[i])
    target.append(3)
for i in range(0,len(f4)-25):
    data.append(f3[i])
    target.append(4)
data = np.array(data)
target = np.array(target)

np.save('sp/data',data)
np.save('sp/target',target)
print(len(data))
print(target)


