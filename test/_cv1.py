from msilib import add_data

import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot  as plt

#from test.test_cv1 import contours

from numpy.ma.extras import average

timg = cv.imread(r"d:\test.jpg",cv.IMREAD_COLOR)
timg = cv.resize(timg,[256,256])
gray_img = cv.cvtColor(timg,cv.COLOR_BGR2GRAY)
gray_img=cv.GaussianBlur(gray_img,(3,3),0)
cv.imshow("origin ",gray_img)
contours = cv.Canny(gray_img.copy(),100,100)
cv.imshow("origin ",contours)
g3adapt= cv.adaptiveThreshold(gray_img,255,\
                  cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,5,2)
#retval,biimg= (
#    cv.threshold(gray_img,130,255,cv.THRESH_BINARY_INV))
cv.imshow("binary ",g3adapt)


contours,hieracy = cv.findContours(
   g3adapt,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#contours = tuple(contour for contour in contours if cv.contourArea(contour)>1)
print(len(contours))
testimg = cv.drawContours(timg.copy(), contours, -1, (255, 0, 0))
cv.imshow("test", testimg)


print(len(contours))# (576,2,1,2)
print(contours[4].shape)
print(contours[0][0][0][1])
#면적이 적은 잡음 제거

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
   xy_data.append([xy_min,xy_max])
print(len(xy_data))
print(xy_data[0])
#마스킹 레이어 안티앨리어싱
for ix in range(len(xy_data)):
    print(type(xy_data[ix]))#numpy
    # 우측은 xy_data[ix][1][0]
    if xy_data[ix][0][0] == 0: continue

    if ix+5>=256:
        break
    cur = 0
    cur_1 = xy_data[ix - 1][0][0] if xy_data[ix - 1] else 300
    cur_2 = xy_data[ix - 2][0][0] if xy_data[ix - 2] else 300
    cur_3 = xy_data[ix - 3][0][0] if xy_data[ix - 3] else 300
    cur_4 = xy_data[ix - 4][0][0] if xy_data[ix - 4] else 300
    cur_5 = xy_data[ix - 5][0][0] if xy_data[ix - 5] else 300
    cur1 = xy_data[ix + 1][0][0] if xy_data[ix + 1] else 300
    cur2 = xy_data[ix + 2][0][0] if xy_data[ix + 2] else 300
    cur3 = xy_data[ix + 3][0][0] if xy_data[ix + 3] else 300
    cur4 = xy_data[ix + 4][0][0] if xy_data[ix + 4] else 300
    cur5 = xy_data[ix + 5][0][0] if xy_data[ix + 5] else 300
    least_low_min = min(cur1,cur2,cur3,cur4,cur5)
    least_top_min = min(cur_1,cur_2,cur_3,cur_4,cur_5)
    if least_top_min>=300:
        cur=0
    else :
        cur = (least_low_min+least_top_min)//2
    xy_data[ix][1][0] =cur
    cur = 0
    cur_1 = xy_data[ix - 1][1][0] if xy_data[ix - 1] else 0
    cur_2 = xy_data[ix - 2][1][0] if xy_data[ix - 2] else 0
    cur_3 = xy_data[ix - 3][1][0] if xy_data[ix - 3] else 0
    cur_4 = xy_data[ix - 4][1][0] if xy_data[ix - 4] else 0
    cur_5 = xy_data[ix - 5][1][0] if xy_data[ix - 5] else 0
    cur1 = xy_data[ix + 1][1][0] if xy_data[ix + 1] else 0
    cur2 = xy_data[ix + 2][1][0] if xy_data[ix + 2] else 0
    cur3 = xy_data[ix + 3][1][0] if xy_data[ix + 3] else 0
    cur4 = xy_data[ix + 4][1][0] if xy_data[ix + 4] else 0
    cur5 = xy_data[ix + 5][1][0] if xy_data[ix + 5] else 0
    least_low_max = max(cur1, cur2, cur3, cur4, cur5)
    least_top_max = max(cur_1, cur_2, cur_3, cur_4, cur_5)
    print(least_low_max)
    print(least_top_max)
    if least_top_max <= 0:
        cur = 0
    else:
        cur = (least_low_min + least_top_min) // 2
    xy_data[ix][0][0] = cur
    cur = 0
    cur_1 = xy_data[ix - 1][1][0] if xy_data[ix - 1] else 0
    cur_2 = xy_data[ix - 2][1][0] if xy_data[ix - 2] else 0
    cur_3 = xy_data[ix - 3][1][0] if xy_data[ix - 3] else 0
    cur_4 = xy_data[ix - 4][1][0] if xy_data[ix - 4] else 0
    cur_5 = xy_data[ix - 5][1][0] if xy_data[ix - 5] else 0
# # (576,2,1,2)
xy_data=np.array(xy_data)#(256,1,2,2)
xy_data= xy_data.reshape(256,1,2,2)
print(xy_data.shape)
extract_img = cv.drawContours(timg.copy(), xy_data, -1, (0, 0, 255))
cv.imshow("ext",extract_img)
print("=======")
cnt=0
new_img = np.zeros((timg.shape))+255
for ix,val,arr in enumerate(timg):#ix = x 좌표
    for iy,val in enumerate(timg[ix]):#[[10 111][200 111]] iy = y 좌표
        if extract_img[ix,0,0,0] <= ix and extract_img[ix,0,1,0] >= ix and extract_img:
            new_img[ix,iy]=timg[ix,iy]
print(new_img.shape)
xy_data_second = (xy_data[:,0,1,:])
print(xy_data_second.shape)
print(xy_data_second.shape)
pts = np.concatenate((xy_data_first,xy_data_second))

pts=pts.reshape((1,-2))
extract_img = cv.drawContours(timg.copy(), pts, -1, (0, 0, 255))
cv.imshow("ext", extract_img)
#cnt+=1
new_img[ix,iy]=timg[ix,iy]
new_img = new_img.astype(np.unit8)
cv.imshow("last_img",new_img)
xy_data_first = xy_data[:,0,0,:]
xy_data_second = (xy_data[:,0,1,:])
print(xy_data_second.shape)
print(xy_data_second.shape)
pts = np.concatenate((xy_data_first,xy_data_second))

pts = pts.reshape((-1,2))
extract_img = cv.drawContours(timg.copy(), pts, -1, (0,0,255))










# #면적순으로 컨투어 정렬
# sorted_contoure =sorted(contours,key=cv.contourArea,reverse=True)
# for i in range(len(sorted_contoure)):
#     contour = sorted_contoure[i]
#     epsilon = 0.01*cv.arcLength(contour,True)
#     approx = cv2.approxPolyDP(contour,epsilon,True)
#     cv.drawContours(timg, [contour], -1, (0, 0, 255))
#     cv.drawContours(timg, [approx], -1, (255, 0, 0))
# cv.imshow("gray", timg)










cv.waitKey(0)
cv.destroyAllWindows()

