import cv2
import cv2 as cv
import numpy as np
timg = cv.imread(r"d:\test.jpg",cv.IMREAD_COLOR)


timg = cv.resize(timg,[256,256])
# cv.imshow("origin",timg)
# bimg = cv.blur(timg,(3,3))
# cv.imshow("blur",bimg)
# g1img = cv.GaussianBlur(timg,(3,3),1)
# cv.imshow("gauss1",g1img)
# g2img = cv.GaussianBlur(timg,(3,3),0.1)
# cv.imshow("gauss2",g2img)
g3img = cv.bilateralFilter(timg,9,75,75)


g3img_gray = cv.cvtColor(g3img,cv.COLOR_RGBA2GRAY)
cv.imshow("bilateral",g3img_gray)
#경계 임계값 - 그레이스케일 변경
# retval1,g3thres = cv.threshold(g3img_gray,127,255,cv.THRESH_BINARY)
# cv.imshow("threshold",g3thres)
# print(g3thres)
# g3adapt= cv.adaptiveThreshold(g3img_gray,255,\
#                      cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,2)
# cv.imshow("adaptThres_mean",g3adapt)
g3adapt= cv.adaptiveThreshold(g3img_gray,255,\
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,15,2)
cv.imshow("adaptThres_gauss",g3adapt)
# retval2,g3otsu= cv.threshold(g3img_gray,0,255,\
#                      cv.THRESH_OTSU)
# cv.imshow("adaptThres_OTSU",g3otsu)
# g_flatten = g3otsu.flatten()
# gmask = g_flatten>=255 #255 보다 크면 True 그렇지 않으면 False
# i_flatten = g3img_gray.flatten() # 그레이 이미지를 평탄화
# ix_np = np.where(gmask) #True의 인덱스 위치값을 반환
# i_flatten[ix_np]=255 # 이진화이미지에서 필터링 된곳 해당 위치의 값을 255(흰색)으로 설정
# cvtimg = i_flatten.reshape(256,256) #이미지 모양으로 변경,
mask=np.where(g3adapt>=255)
timg[mask]=255
cv.imshow("filterimg",timg)#필터링 된 이미지 확인
# 1. 이미지 불러오기
# 2. 블러처리로 노이즈제거
# 3. 그레이스케일로 변경
# 4. 이미지 이진화처리(threshold, adaptthreshold, ...)
# 5. 이미지 경계값 출력
# 6. 이미지 경계선 그리기
# 7. 경계선 안쪽 이미지 추출
contours, hierarchy=cv.findContours(
   g3adapt, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_NONE)
print(len(contours))
print(contours[0])
print(hierarchy.shape)
cont_img = cv.drawContours(timg.copy(),contours,-1,(0,0,255),1)
cv.imshow("contour img",cont_img)








cv.waitKey(0)
cv.destroyAllWindows()

