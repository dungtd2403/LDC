import cv2
import numpy as np

import glob
from glob import glob

# img = cv2.imread("/home/dung/Project/LDC/data/1.JPG")
# a = np.average(img, axis=(0,1))

def glob_img_mean(images_dir):
    mean_list =[]
    # count = 0
    # mean = [0,0,0]
    images = sorted(glob(f"{images_dir}/*.JPG"))
    for img in images:
        # print(img)
        image = cv2.imread(f'{img}')
        # print(img.shape)
        mean = np.average(image, axis=(0,1))
        # mean += mean
        mean_list.append(mean)
        # count += 1
        mean_all = sum(mean_list) / len(mean_list)
    return mean_all
if __name__ == "__main__":
    img_dir = '/home/dung/DL_Project/LDC/sem_data/SEM_img/train'
    mean = glob_img_mean(img_dir)
    print(mean)
    print(mean.shape)