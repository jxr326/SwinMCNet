#coding=utf-8
import os
import cv2
import numpy as np




def split_map(datapath):
    print(datapath)
    for name in os.listdir(datapath+'/GT/'):
        mask = cv2.imread(datapath+'/GT/'+name,0)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(datapath+'/skeleton/'):
            os.makedirs(datapath+'/skeleton/')
        cv2.imwrite(datapath+'/skeleton/'+name, body)

        if not os.path.exists(datapath+'/contour/'):
            os.makedirs(datapath+'/contour/')
        cv2.imwrite(datapath+'/contour/'+name, mask-body)


if __name__=='__main__':
    split_map('../dataset')
