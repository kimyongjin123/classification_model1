import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot  as plt

from test.test_cv1 import xy_data

timg = cv.imread(r"d:\test2.jpg",cv.IMREAD_COLOR)
#timg = cv.imread(r"d:\aimodel.jpg",cv.IMREAD_COLOR)
timg = cv.resize(timg,[256,256])
gray_img = cv.cvtColor(timg,cv.COLOR_BGR2GRAY)
gray_img=cv.GaussianBlur(gray_img,(3,3),0)
cv.imshow("origin ",gray_img)
# g3adapt = cv.Canny(gray_img.copy(),100,100)
g3adapt= cv.adaptiveThreshold(gray_img,255,\
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,5,2)
# retval,g3adapt= (
#     cv.threshold(gray_img,127.5,255,cv.THRESH_BINARY_INV))
# cv.imshow("binary ",g3adapt)


contours,hieracy = cv.findContours(
   g3adapt,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
contours = tuple(contour for contour in contours  if cv.contourArea(contour)>1)
print(len(contours))
testimg = cv.drawContours(timg.copy(), contours, -1, (255, 0, 0))
cv.imshow("test", testimg)


print(len(contours))# (576,2,1,2)
xy_data = np.array(xy_data)# (
# print(contours[4].shape)
# print(contours[0][0][0][1])
#면적이 적은 잡음 제거


xy_data=[]
pre_min = [];pre_max=[]
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
   if xy_min[0]==300 :# 추출이 안되고 있는곳 처리
       if len(pre_min)>0:
           xy_min = pre_min
           pre_min = []
       else:
           xy_min = np.array([-1,-1])

   if xy_max[0]<0 :
       if len(pre_min)>0:
            xy_max= pre_max
            pre_max = []
   else:
       xy_max = pre_max
       pre_max = []
   if xy_min[0]>0 and xy_max[0]<300:
       if ix == 5: print(xy_min, xy_max)
       pre_min = xy_min
       pre_max = xy_max
   if ix == 5:print(xy_min,xy_max)
   xy_data.append([xy_min,xy_max])
#print(len(xy_data))
#print(xy_data[:10])
   if xy_max[0]<0:
       xy_max=np.array([300,300])
   xy_data.append([xy_min,xy_max])
#print(len(xy_data))
#print(xy_data[:10])
#마스킹 레이어 안티앨리어싱
for ix in range(len(xy_data)):
   #print(type(xy_data[ix]))
   # 우측은 xy_data[ix][1][0]
   if xy_data[ix][0][0] < 0 xy_data[ix][1][0] < 0 :
   cur_1 = xy_data[ix - 1][0][0] if ix>=1 and xy_data[ix - 1][0][0]>=0 else 300
   cur_2 = xy_data[ix - 2][0][0] if ix>=2 and xy_data[ix - 2][0][0]>=0 else 300
   cur_3 = xy_data[ix - 3][0][0] if ix>=3 and xy_data[ix - 3][0][0]>=0 else 300
   cur_4 = xy_data[ix - 4][0][0] if ix>=4 and xy_data[ix - 4][0][0]>=0 else 300
   cur_5 = xy_data[ix - 5][0][0] if ix>=5 and xy_data[ix - 5][0][0]>=0 else 300
   cur1=xy_data[ix + 1][0][0] if ix <=254 and xy_data[ix + 1][0][0]>=0 else 300
   cur2=xy_data[ix + 2][0][0] if ix <=253 and xy_data[ix + 2][0][0]>=0 else 300
   cur3=xy_data[ix + 3][0][0] if ix <=252 and xy_data[ix + 3][0][0]>=0 else 300
   cur4=xy_data[ix + 4][0][0] if ix <=251 and xy_data[ix + 4][0][0]>=0 else 300
   cur5=xy_data[ix + 5][0][0] if ix <=250 and xy_data[ix + 5][0][0]>=0 else 300
   #[array([121,6], dtype=int32), array([121, 6], dtype=int32)
   least_low_min = min(cur1,cur2,cur3,cur4,cur5)
   least_top_min = min(cur_1, cur_2, cur_3, cur_4, cur_5)
   if least_low_min<0 or least_top_min<0 or least_low_min>=300 or least_top_min>=300 :
       print("xxxxxxx",ix)
       continue
   if least_low_min>=300 and least_top_min<300:
       cur = least_top_min
   elif least_top_min>=300 and least_low_min<300:
       cur = least_low_min
   else:
       cur = (least_low_min+least_top_min)//2
   xy_data[ix][0][0]=cur
   cur = 0
   cur_1 = xy_data[ix - 1][1][0] if ix>=1 and xy_data[ix - 1][1][0]<300 else -1
   cur_2 = xy_data[ix - 2][1][0] if ix>=2 and xy_data[ix - 2][1][0]<300 else -1
   cur_3 = xy_data[ix - 3][1][0] if ix>=3 and xy_data[ix - 3][1][0]<300 else -1
   cur_4 = xy_data[ix - 4][1][0] if ix>=4 and xy_data[ix - 4][1][0]<300 else -1
   cur_5 = xy_data[ix - 5][1][0] if ix>=5 and xy_data[ix - 5][1][0]<300 else -1
   cur1 = xy_data[ix + 1][1][0] if ix <=254 and xy_data[ix + 1][1][0]<300 else -1
   cur2 = xy_data[ix + 2][1][0] if ix <=253 and xy_data[ix + 2][1][0]<300 else -1
   cur3 = xy_data[ix + 3][1][0] if ix <=252 and xy_data[ix + 3][1][0]<300 else -1
   cur4 = xy_data[ix + 4][1][0] if ix <=251 and xy_data[ix + 4][1][0]<300 else -1
   cur5 = xy_data[ix + 5][1][0] if ix <=250 and xy_data[ix + 5][1][0]<300 else -1
   least_low_min = min(cur1, cur2, cur3, cur4, cur5)
   least_top_min = min(cur_1, cur_2, cur_3, cur_4, cur_5)
   if ix==4 : print(cur1,cur2,cur3,cur4,cur5,cur_1,cur_2,cur_3,cur_4,cur_5)
   if least_low_min>=300 and least_top_min>300:
       cur = least_top_min
   elif least_top_min>=300:

       cur = (least_low_max + least_top_max) // 2
   xy_data[ix][1][0] = cur
for ix in range(len(xy_data)):
   #print(type(xy_data[ix]))
   # 우측은 xy_data[ix][1][0]
   if xy_data[ix][0][0] < 0 or xy_data[ix][1][0] < 0 :continue
   cur = 0
   cur_1 = xy_data[ix - 1][0][0] if ix>=1 and xy_data[ix - 1][0][0]>=0 else 300
   cur_2 = xy_data[ix - 2][0][0] if ix>=2 and xy_data[ix - 2][0][0]>=0 else 300
   cur_3 = xy_data[ix - 3][0][0] if ix>=3 and xy_data[ix - 3][0][0]>=0 else 300
   cur_4 = xy_data[ix - 4][0][0] if ix>=4 and xy_data[ix - 4][0][0]>=0 else 300
   cur_5 = xy_data[ix - 5][0][0] if ix>=5 and xy_data[ix - 5][0][0]>=0 else 300
   cur1=xy_data[ix + 1][0][0] if ix <=254 and xy_data[ix + 1][0][0]>=0 else 300
   cur2=xy_data[ix + 2][0][0] if ix <=253 and xy_data[ix + 2][0][0]>=0 else 300
   cur3=xy_data[ix + 3][0][0] if ix <=252 and xy_data[ix + 3][0][0]>=0 else 300
   cur4=xy_data[ix + 4][0][0] if ix <=251 and xy_data[ix + 4][0][0]>=0 else 300
   cur5=xy_data[ix + 5][0][0] if ix <=250 and xy_data[ix + 5][0][0]>=0 else 300
   least_low_min = min(cur1,cur2,cur3,cur4,cur5)
   least_top_min = min(cur_1, cur_2, cur_3, cur_4, cur_5)
   if least_low_min<0 or least_top_min<0 or least_low_min>=300 or least_top_min>=300 :
       continue
   if least_low_min>=300 and least_top_min<300:
       cur = least_top_min
   elif least_top_min>=300 and least_low_min<300:
       cur = least_low_min
   else:
       cur = (least_low_min+least_top_min)//2


   xy_data[ix][0][0]=cur
   cur = 0
   cur_1 = xy_data[ix - 1][1][0] if ix>=1 and xy_data[ix - 1][1][0]<300 else -1
   cur_2 = xy_data[ix - 2][1][0] if ix>=2 and xy_data[ix - 2][1][0]<300 else -1
   cur_3 = xy_data[ix - 3][1][0] if ix>=3 and xy_data[ix - 3][1][0]<300 else -1
   cur_4 = xy_data[ix - 4][1][0] if ix>=4 and xy_data[ix - 4][1][0]<300 else -1
   cur_5 = xy_data[ix - 5][1][0] if ix>=5 and xy_data[ix - 5][1][0]<300 else -1
   cur1 = xy_data[ix + 1][1][0] if ix <=254 and xy_data[ix + 1][1][0]<300 else -1
   cur2 = xy_data[ix + 2][1][0] if ix <=253 and xy_data[ix + 2][1][0]<300 else -1
   cur3 = xy_data[ix + 3][1][0] if ix <=252 and xy_data[ix + 3][1][0]<300 else -1
   cur4 = xy_data[ix + 4][1][0] if ix <=251 and xy_data[ix + 4][1][0]<300 else -1
   cur5 = xy_data[ix + 5][1][0] if ix <=250 and xy_data[ix + 5][1][0]<300 else -1
   least_low_min = min(cur1, cur2, cur3, cur4, cur5)
   least_top_min = min(cur_1, cur_2, cur_3, cur_4, cur_5)
   if least_low_max<0 and least_top_max>=0:
       cur = least_top_max
   elif least_top_max<0 and least_low_max>=0:
       cur = least_low_max
   else:
       cur = (least_low_max + least_top_max) // 2
   xy_data[ix][1][0] = cur




# # (576,2,1,2)
xy_data=np.array(xy_data)#(256,1,2,2)
xy_data= xy_data.reshape(256,1,2,2)
print(xy_data.shape)
extract_img = cv.drawContours(timg.copy(), xy_data, -1, (0, 0, 255))
cv.imshow("ext",extract_img)
print("===========")
cnt=0
new_img = np.zeros((timg.shape))+255
for ix,valarr in enumerate(timg):#ix = x 좌표
   for iy,val in enumerate(timg[ix]):#[[10 111][200 111]] iy = y 좌표
       # print(xy_data.shape)
       # print(xy_data[ix,0])
       if xy_data[ix,0,0,0] <= iy and xy_data[ix,0,1,0] >= iy\
               and xy_data[ix,0,0,1]==ix:
           cnt+=1
           new_img[ix,iy]=timg[ix,iy]
new_img = new_img.astype(np.uint8)
cv.imshow("last_img",new_img)
cv.imwrite(r"d:\test.jpg")












# xy_data_first = xy_data[:,0,0,:]
# xy_data_second = (xy_data[:,0,1,:])
# print(xy_data_second.shape)
# print(xy_data_second.shape)
# pts = np.concatenate((xy_data_first,xy_data_second))
#
# pts=pts.reshape((-1,2))
# extract_img = cv.drawContours(timg.copy(), pts, -1, (0, 0, 255))
# cv.imshow("ext",extract_img)
















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



