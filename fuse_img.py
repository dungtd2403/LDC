import cv2 
import os
import numpy as np

img = cv2.imread("/home/dung/Seg_tel/LDC/result/BRIND2CLASSIC/fused/1.png", cv2.COLOR_BGR2GRAY)  # Read image
img_2 = cv2.imread("/home/dung/Seg_tel/LDC/data/1.JPG")

# Setting parameter values
t_lower = 100  # Lower Threshold
t_upper = 200  # Upper threshold
# aperture_size = 5
L2Gradient = True
# Applying the Canny Edge filter

# draw contour
thresh = 100
ret,thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

edge = cv2.Canny(thresh_img, t_lower, t_upper, L2gradient = L2Gradient )

contours, hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
# contours_np = np.array(contours)
print(len(contours))
img_contours = np.zeros(img.shape)
img_contours.fill(255)
# fill_contour = cv2.fillPoly(img_contours, pts =contours_np, color=(255,255,255))
# print(fill_contour)

# img_contours = np.zeros(img.shape)
for con in range(len(contours)):
    # print(len(contours[con]))
    
    if len(contours[con]) > 100:
        print('area',cv2.contourArea(contours[con]))
        print('arclength',cv2.arcLength(contours[con], True))
        img_contours = cv2.drawContours(img_contours, contours ,con, color=(0, 0, 0), thickness=3)
        cv2.waitKey(0)
        cv2.imshow('test', img_contours)
    else: 
        continue
# img_contours = cv2.drawContours(img_2, fill_contour, -1, (0,255,0), 3)
# cv2.imwrite('./contours.png',img_contours)
 
# cv2.imshow('original', img_contours)
# cv2.imshow('edge', img_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# if __name__ == "__main__"