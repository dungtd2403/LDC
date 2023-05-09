import numpy as np
import os
import cv2
import glob
from glob import glob
from os import path
def sp_noise(image, prob, mode):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        if mode =='s':
            white = 255       
        if mode =='p':
            black = 0 
        if mode =='sap':
            white =255
            black= 0
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            if mode =='s':
                white = np.array([255, 255, 255], dtype='uint8')
            if mode =='p':
                black = np.array([0, 0, 0], dtype='uint8')
            if mode =='sap':
                white = np.array([255, 255, 255], dtype='uint8')
                black = np.array([0, 0, 0], dtype='uint8')
        else:  # RGBA
            if mode =='s':
                white = np.array([255, 255, 255,255], dtype='uint8')
            if mode =='p':
                black = np.array([0, 0, 0,0], dtype='uint8')
            if mode =='sap':
                white = np.array([255, 255, 255,255], dtype='uint8')
                black = np.array([0, 0, 0,0], dtype='uint8')
    if mode =='s':
        probs = np.random.random(output.shape[:2])   
        output[probs > 1 - (prob / 2)] = white   
    if mode =='p':
        probs = np.random.random(output.shape[:2])
        output[probs < (prob / 2)] = black
    if mode =='sap':
        probs = np.random.random(output.shape[:2])
        output[probs < (prob / 2)] = black
        output[probs > 1 - (prob / 2)] = white
        
    return output

def add_gaussian_noise(img, var):
    # gaussian_noise_imgs = []
    # row, col = X_imgs.shape
    # Gaussian distribution parameters
    mean = 0 #101.34525247
    sigma = var ** 0.5
    
    gaussian = np.random.normal(mean, sigma, (960, 1280)) #  np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian 
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image

# noise = np.random.poisson(10)
# noise_img = img + noise
# noise_img = 255 * (noise_img / np.amax(noise_img))
# noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
def add_poisson_noise(img, sigma):
    # noise = np.random.poisson(sigma)
    # ps_img = img + noise
    # # ps_img = cv2.add(img,noise)
    # ps_img  = 255 * (ps_img  / np.amax(ps_img ))
    # ps_img = np.clip(ps_img , 0, 255).astype(np.uint8)
    # noise_img = 255 * 
    # noise_img[:,:,0] = np.ones([960,1280])*64
    # noise_img[:,:,1] = np.ones([960,1280])*128
    # noise_img[:,:,2] = np.ones([960,1280])*192
    # noise = np.array(ps_img , dtype=np.uint8)
    # print(noise_img.shape)
    # image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # Add Poisson noise
    noise = np.random.poisson(img)
    noisy_img = np.clip(noise, 0, 255).astype(np.uint8)
    return noisy_img 
def add_uniform_noise(img, weight=0.5):
    uni_noise=np.zeros((960,1280),dtype=np.uint8)
    cv2.randu(uni_noise,0,255)
    uni_noise=(uni_noise*weight).astype(np.uint8)
    un_img=cv2.add(img,uni_noise)
    return un_img


if __name__=="__main__":
    images_dir = '/home/dung/dataset/SEM_TEL/SEM_100/SEM_img'
    name = glob(f"{images_dir}/*.JPG")
    des_dir = '/home/dung/Project/LDC/noise_data/poisson_fix'
    names = [str(path.basename(i)).replace(".JPG", "") for i in name]
    # print(names)
    for img_name in names:
        img_path = images_dir + '/' + f'{img_name}.JPG'
        print(img_path)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        print(img.shape)    
        # noise = add_gaussian_noise(img, 2304) # 144, 1024, 2304
        noise = add_poisson_noise(img,30)
        # noise = add_uniform_noise(img,0.8)
        # noise = sp_noise(img,0.4,mode= 'sap')
        # im_o = cv2.hconcat([noise, img])
        # cv2.imshow('test',noise)
        cv2.imwrite(f'{des_dir}/{img_name}.JPG',noise)
        # cv2.waitKey(0)
        # break