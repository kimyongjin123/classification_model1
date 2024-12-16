import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot  as plt

from imageCrawling import cnt_count
from test_cv import new_img

timg = cv.imread(r"d:\test.jpg",cv.IMREAD_COLOR)
timg = cv.resize(timg)
gray_img = cv.cvtColor(timg,cv.COLOR_BGR2GRAY)
gray_img=cv.GaussianBlur(gray_img,(1,1),0)
cv.imshow("origin ",gray_img)
# contours = cv.Canny(gray_img.copy(),100,100)
# cv.imshow("origin ",contours)
g3adapt= cv.adaptiveThreshold(gray_img,255,\
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,5,2)
# retval,biimg= (
#     cv.threshold(gray_img,130,255,cv.THRESH_BINARY_INV))
cv.imshow("binary ",g3adapt)


contours,hieracy = cv.findContours(
   g3adapt,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
testimg = cv.drawContours(timg.copy(), contours, -1, (255, 0, 0))
cv.imshow("test", testimg)


print(len(contours))# (576,2,1,2)
print(contours[4].shape)
print(contours[0][0][0][1])
xy_data=[]
for ix in range(256):
   xy_min = [300,300];xy_max=[0,0]
   for i,data in enumerate(contours):
       for n,indata in enumerate(data):
           #print(indata[0][1])#y  위치
           if indata[0][1]==ix:
               if xy_min[0]>indata[0][0]:
                   xy_min=indata[0]
               if xy_max[0]<indata[0][0]:
                   xy_max = indata[0]
   if xy_min[0]==300:# 추출이 안되고 있는곳 처리
       xy_min=[0,0]
   xy_data.append([xy_min,xy_max])
print(len(xy_data))
print(xy_data[0])
# 마스킹 레이어 안티앨리어싱
# (576,2,1,2)
xy_data=np.array(xy_data)#(256,1,2,2)
xy_data= xy_data.reshape(256,1,2,2)
print(xy_data.shape)
extract_img = cv.drawContours(timg.copy(), xy_data, -1, (0, 0, 255))
cv.imshow("ext",extract_img)
print("==========")
print(xy_data.shape)

new_img = np.zeros((timg.shape))+255
for ix,valarr in enumerate(timg):#ix = x 좌표
    # print(xy_data.shape)
    # print(xy_data[ix,0])
        print("xxxxxxxxx")
        print(xy_data[ix,0,0,0])
        print(xy_data[ix,0,1,0])
        if xy_data[ix,0,0,0] <= iy and xy_data[ix,0,1,0] >= ix\
                and xy_data[ix,0,0,1]==ix:
            cnt+=1
            new_img[ix,iy]=timg[ix,iy]
new_img = new_img.astype(np.uint8)
cv.imshow("last_img",new_img)
xy_data_first = xy_data[:,0,0,:]
xy_data_second = (xy_data[:,0,1,:])
print(xy_data_second.shape)
print(xy_data_second.shape)
pts = np.concatenate((xy_data_first,xy_data_second))

pts=pts.reshape((-1,2))
hull = cv.convexHull(contours)
extract_img = cv.drawContours(timg.copy(), hull, -1, (0, 0, 255))
cv.imshow("ext",extract_img)
