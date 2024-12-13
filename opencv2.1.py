import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot  as plt
timg = cv.imread(r"d:\test.jpg",cv.IMREAD_COLOR)
timg = cv.resize(timg,[256,256])
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
   xy_data.append([xy_min,xy_max])
print(len(xy_data))
print(xy_data[0])
# (576,2,1,2)
xy_data=np.array(xy_data)#(256,1,2,2)
xy_data= xy_data.reshape(256,1,2,2)
print(xy_data.shape)
extract_img = cv.drawContours(timg.copy(), xy_data, -1, (0, 0, 255))
cv.imshow("ext",extract_img)














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

