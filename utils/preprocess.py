import os
import cv2
import numpy as np
import torch
import kornia as kn

def constrast():
    img = cv2.imread('/home/dung/Project/LDC/result/SEM_b=42CLASSIC/fused/33.png', 1)
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    # result = np.hstack((img, enhanced_img))
    # cv2.imshow('enhanced_img',enhanced_img) 
    # cv2.imshow('original_img',img)
    # cv2.imwrite('test_img.png',enhanced_img)
    # cv2.waitKey(0)
def convert_rgb_to_yuv(frame):
    """
    Convert a given rgb image into hsv image
    :param frame: Color image to convert
    :return: YUV image as numpy array
    """

    # CODE HERE

    #Conversion matrix from rgb to yuv, transpose matrix is used to convert from yuv to rgb
    yuv_from_rgb = np.array([[0.114, 0.587,  0.299],
                            [0.436, -0.28886, -0.14713],
                            [-0.10001, -0.51499, 0.615]])

    # do conversion
    image = frame.dot(yuv_from_rgb.T) 
    # add the constants based on the conversion formula
    image += np.array([16, 128, 128]).reshape(1, 1, 3)
    # convert the image to uint8 format
    image = np.array(image, dtype = "uint8")
    return image
def contour_extractor(mask_path,img_path):
    img_mask = cv2.imread(f"{mask_path}", cv2.COLOR_BGR2GRAY)  # Read image
    # break
    # cv2.waitKey(0)
    # kernel = np.ones((9,9), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img = cv2.imread(f"{img_path}")
    original = cv2.imread(f"{img_path}")
    # print(img_mask)
    thresh = 5
    print(img_mask.shape)
    kernels = np.array([[-1,-1,-1], [-1,20,-1], [-1,-1,-1]])
# Apply the sharpening kernel to the image using filter2D
    sharpened = cv2.filter2D(img_mask, -1, kernels)
    cv2.imshow('sharpened',sharpened)
    cv2.waitKey(0)
    ret,thresh_img = cv2.threshold(sharpened, thresh, 255, cv2.THRESH_BINARY_INV)
    # print(thresh_img)
    print(thresh_img.shape)
    cv2.imshow('thresh_img',thresh_img)
    cv2.waitKey(0)
    # erode_mask = cv2.erode(thresh_img, kernel, iterations=1)
    # opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    # print(opening.shape)
    # cv2.imshow('erode_mask',opening)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
        # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST,mode)
    # bilateral = cv2.bilateralFilter(thresh_img ,9,75,75)
    # cv2.imshow('dilate',bilateral )
    # cv2.waitKey(0)
    
    for con in range(len(contours)):
        # if len(contours[con]) > con_thresh and cv2.contourArea(contours[con]) > cv2.arcLength(contours[con], True) and len(contours[con]) < 2000 :
        if len(contours[con]) > 300:
            print('area',cv2.contourArea(contours[con]))
            print(len(contours[con]))
            print('arclength',cv2.arcLength(contours[con], True))
            
            img_contours = cv2.drawContours(img, contours ,con, color=(255, 0, 0), thickness=1)
            cv2.imshow('test', img_contours)
            cv2.waitKey(0)

        else: 
            continue
    im_o = cv2.hconcat([original, img])
    # print(save_dir_setting)
    # save_path = save_dir_setting  + '/' + m 
    # cv2.imwrite(save_path, im_o)

if __name__ =='__main__':
    
    mask_path = '/home/dung/Project/LDC/utils/test_img.png'
    img_path = '/home/dung/Project/LDC/result/SEM_b=42CLASSIC/fused/33.png'
    contour_extractor(mask_path=mask_path, img_path=img_path)