import cv2 
import os
import numpy as np
import glob
from glob import glob
import os

# edge = cv2.dilate(img, kernel)
# img[img < 250] = 0
# img[img >= 250] = 255


###### canny 
# t_lower = 100  # Lower Threshold
# t_upper = 200  # Upper threshold
# edge = cv2.Canny(thresh_img, t_lower, t_upper, L2gradient = L2Gradient )
# fill_contour = cv2.fillPoly(img_contours, pts =contours_np, color=(255,255,255))
# print(fill_contour)

# kernel2 = np.ones((3,3), np.uint8)
# L2Gradient = True

# ksize = (5, 5)
# image = cv2.blur(img, ksize) 
# cv2.waitKey(0)
# cv2.imshow('image', image)

# draw contour
# cv2.waitKey(0)
# cv2.imshow('bilateral', bilateral)


# bilateral = cv2.bilateralFilter(thresh_img ,9,75,75)
# eroded_mask = cv2.erode(thresh_img, kernel2, iterations=1)
# dilate_mask = cv2.dilate(thresh_img, kernel1, iterations=1)

# cv2.imshow('thresh_img',thresh_img)
# cv2.waitKey(0)
# cv2.imshow('dilate_mask',dilate_mask)
# cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img_contours = cv2.drawContours(img_2, contours ,-1, color=(255, 0, 0), thickness=1)
# contours, hierarchy = cv2.findContours(dilate_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# contours_np = np.array(contours)
# print(len(contours))
# img_contours = np.zeros(img.shape)
# img_contours.fill(255)




# for con in range(len(contours)):
#     if len(contours[con]) > 400  and cv2.contourArea(contours[con]) > cv2.arcLength(contours[con], True):
#     # if len(contours[con]) > 200:
#         print('area',cv2.contourArea(contours[con]))
#         print('arclength',cv2.arcLength(contours[con], True))
        
#         img_contours = cv2.drawContours(img_2, contours ,con, color=(255, 0, 0), thickness=3)
        
        
#         # cv2.imshow('test', edge)
#         cv2.imshow('test', img_contours)
#         cv2.waitKey(0)

#     else: 
#         continue
# # img_contours = cv2.drawContours(img_2, fill_contour, -1, (0,255,0), 3)
# cv2.imwrite('./contours.png',img_contours)
 
# cv2.imshow('original', img_contours)
# cv2.imshow('edge', img_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
def fusemask(mask_dir,images_dir, save_dir, kernel, setting):
    save_dir_setting = save_dir + '/' + setting
    # os.mkdir(save_dir_setting)
    for i,m in enumerate(os.listdir(mask_dir)):
        print(m)
        mask_path = mask_dir + '/' + m
        img_path = images_dir + '/' + m[:-4] + '.JPG'
        # print(img_path)
        # print(mask_path)
        img_mask = cv2.imread(f"{mask_path}", cv2.COLOR_BGR2GRAY)  # Read image
        # break
        img = cv2.imread(f"{img_path}")
        original = cv2.imread(f"{img_path}")
        # print(img_mask)
        img_mask[img_mask < 220] = 0
        img_mask[img_mask >= 220] = 255
        # edge_map[edge_map < 100] = 0
        # edge_map[edge_map >= 100] = 255
        dilate_mask = cv2.dilate(img_mask, kernel, iterations=1)
        height, width  = img_mask.shape
        for i in range(height):
            for j in range(width):
                # Get the pixel values at (i, j)
                # print(edge_map[i,j])
                if dilate_mask[i,j] == 0:
                    img[i,j,0] = 255
                    img[i,j,1] = 0
                    img[i,j,2] = 0
        im_o = cv2.hconcat([original, img])
        print(save_dir_setting)
        save_path = save_dir_setting  + '/' + m 
        # cv2.imwrite(save_path, im_o)
        cv2.imshow('dilate',img)
        cv2.waitKey(0)
        # dilate_mask = cv2.dilate(thresh_img, kernel, iterations=1)
        # contours, hierarchy = cv2.findContours(dilate_mask, cv2.RETR_LIST,mode)


def post_processing(mask_dir,images_dir, save_dir, thresh, kernel, setting, line_width, mode,con_thresh):
    
    save_dir_setting = save_dir + '/' + setting
    # os.mkdir(save_dir_setting)
    for i,m in enumerate(os.listdir(mask_dir)):
        print(m)
        mask_path = mask_dir + '/' + m
        img_path = images_dir + '/' + m[:-4] + '.JPG'
        # print(img_path)
        # print(mask_path)
        img_mask = cv2.imread(f"{mask_path}", cv2.COLOR_BGR2GRAY)  # Read image
        # break
        img = cv2.imread(f"{img_path}")
        original = cv2.imread(f"{img_path}")
        # print(img_mask)
        ret,thresh_img = cv2.threshold(img_mask, thresh, 255, cv2.THRESH_BINARY)
        # print(thresh_img)
        start_point = 0,0 
        end_point = 1279, 959
        color = (0,0,0)
        thresh_img = cv2.rectangle(thresh_img, start_point, end_point, color, 1)
        dilate_mask = cv2.dilate(thresh_img, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilate_mask, cv2.RETR_EXTERNAL,mode)
        
            # contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST,mode)
        # bilateral = cv2.bilateralFilter(thresh_img ,9,75,75)
        # cv2.imshow('dilate',bilateral )
        # cv2.waitKey(0)
        
        for con in range(len(contours)):
            if len(contours[con]) > con_thresh and cv2.contourArea(contours[con]) > cv2.arcLength(contours[con], True) and len(contours[con]) < 2000 :
            # if len(contours[con]) > 300:
                print('area',cv2.contourArea(contours[con]))
                print(len(contours[con]))
                print('arclength',cv2.arcLength(contours[con], True))
                
                img_contours = cv2.drawContours(img, contours ,con, color=(255, 0, 0), thickness=line_width)
                cv2.imshow('test', img_contours)
                cv2.waitKey(0)

            else: 
                continue
        im_o = cv2.hconcat([original, img])
        print(save_dir_setting)
        save_path = save_dir_setting  + '/' + m 
        # cv2.imshow('test', im_o)
        # cv2.waitKey(0)
        # cv2.imwrite(save_path, im_o)


if __name__ == "__main__":
    # kernel_cus = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,255,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    kernels1 = [np.ones((3,3), np.uint8), np.ones((4,4), np.uint8), np.ones((5,5), np.uint8)]
    model = 'B4Sem' # b4SEM 
    modes = [cv2.CHAIN_APPROX_NONE , cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS]
    thresh = 250  
    thickness= [1,2]
    con_threshs = [300,400,500] 
    
    mask_dir = "/home/dung/Project/LDC/result_stage1/check_0805/B8_720_960_m60/fused"
    image_dir = "/home/dung/Project/LDC/data"
    save_dir ="/home/dung/Project/LDC/demo_500"
    # for line_width in thickness:
    #     for con_thresh in con_threshs:
    #         for mode in modes:
    #             for i,kernel1 in enumerate(kernels1):
    #                 kernel1_shape =  kernel1.shape
    #                 setting = f'{model}_thickness{line_width}_kernelshape{kernel1_shape}_mode{mode}_{con_thresh}'
    #                 post_processing(mask_dir=mask_dir, images_dir= image_dir,save_dir= save_dir, thresh=thresh, kernel= kernel1,setting=setting,line_width= line_width, mode= mode, con_thresh =con_thresh)
    
    line_width = 1
    kernel1 = np.ones((3,3), np.uint8)
    mode = cv2.CHAIN_APPROX_NONE
    con_thresh = 300
    # setting = f'{model}_thickness{line_width}_kernelshape{kernel1.shape}_mode{mode}_{con_thresh}_no_circle_loop'
    setting = f'{model}_kernelshape{kernel1.shape}_fuse_directly_thresh250'
    # setting = f'{model}_thickness{line_width}_nodilate_mode{mode}_{con_thresh}_no_circle_loop'
    # post_processing(mask_dir=mask_dir, images_dir= image_dir,save_dir= save_dir, thresh=thresh, kernel= kernel1,setting=setting,line_width= 1, mode= cv2.CHAIN_APPROX_TC89_L1, con_thresh = con_thresh)
    fusemask(mask_dir=mask_dir, images_dir= image_dir,save_dir= save_dir, kernel= kernel1,setting=setting)