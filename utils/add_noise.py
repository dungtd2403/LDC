import numpy as np
import os
import cv2
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

def add_gaussian_noise(X_imgs):
    # gaussian_noise_imgs = []
    row, col, _ = X_imgs.shape
    # Gaussian distribution parameters
    mean = 101.34525247
    var = 1024
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

if __name__=="__main__":
    img_path = '/home/dung/Project/LDC/sem_data/SEM_img/train/2.JPG'
    img = cv2.imread(img_path)
    noise = add_gaussian_noise(img)
    # noise = sp_noise(img,0.5,mode= 's')
    # im_o = cv2.hconcat([noise, img])
    cv2.imshow('test',noise)
    cv2.waitKey(0)
    # gaussian = np.random.normal(mean, sigma, (224, 224)) #  np.zeros((224, 224), np.float32)

    # noisy_image = np.zeros(img.shape, np.float32)

    # if len(img.shape) == 2:
    #     noisy_image = img + gaussian
    # else:
    #     noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    #     noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    #     noisy_image[:, :, 2] = img[:, :, 2] + gaussian

    # cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    # noisy_image = noisy_image.astype(np.uint8)