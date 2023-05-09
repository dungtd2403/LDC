import cv2
import numpy as np 

img_path = '/home/dung/Project/LDC/result/SEM_b=12CLASSIC/fused/46.png'
img = cv2.imread(f"{img_path}")
kernel = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,255,0,0],[0,0,0,0,0],[0,0,0,0,0]])
print(kernel)
image = cv2.filter2D(img, -1, kernel)
cv2.imshow('img_conv',image)
cv2.waitKey(0)