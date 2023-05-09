import shutil
import os
import glob
from glob import glob
from os import path
# image = [51,54,55, 60, 64, 65, 68 , 70, 71 , 73 , 82 , 83 , 90 , 92 , 96 , 98, 109 ,124,128, 131, 136, 138, 140 ,143, 145 , 166 , 168, 173, 176,184, 186 , 190 , 213,214 , 215 , 226, 227,232 ,240 , 241, 244 , 245, 246, 250, 295, 309, 314 ,343,344, 351, 365 , 368 , 394, 396 ,448 ,468, 470, 471, 482, 484, 495 ]
# print(len(image))

image_dir = '/home/dung/Project/LDC/sem_data/SEM_img/train'
images = sorted(glob(f"{image_dir}/*.JPG"))
# print(images)
names = [str(path.basename(i)).replace(".JPG", "") for i in images]
# print(names)
avg_dir = '/home/dung/Project/LDC/result_stage1/check_0805/B8_720_960_m60/avg'

for i in range(len(names)):
    img_path = avg_dir + '/' + str(names[i]) + '.png'
    # print(img_path)
    des_path = '/home/dung/Project/LDC/sem_data/SEM_stage2_new/train' 
    shutil.copy(img_path, des_path)