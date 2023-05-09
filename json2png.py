import os
import json
from os import path
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image, ImageDraw


def gen_mask(images_dir, label_dir, output_dir):
    print(images_dir)
    print(label_dir)
    val_list = ["1.json","4.json","11.json","14.json","19.json","21.json","25.json","33.json","39.json","46.json"]
    # images = sorted(glob(f"{images_dir}/*.JPG") + glob(f"{images_dir}/*.png"))
    images = sorted(glob(f"{images_dir}/*.JPG") + glob(f"{images_dir}/*.png"))
    labels = sorted(glob(f"{label_dir}/*.json"))
    print(labels)

    # assert images and labels and len(images) == len(labels)

    if not path.exists(output_dir):
        os.mkdir(output_dir)

    names = [str(path.basename(i)).replace(".json", "") for i in labels]

    for idx, l in tqdm(enumerate(labels)):
        data = json.load(open(l, 'r', encoding='utf8'))
        o_img = np.zeros((data['imageHeight'], data['imageWidth']), dtype='uint8')
        for shape in data['shapes']:
            points = np.array(shape['points'], dtype='int32')
            print(points)
            for p in range(len(points)):
                cv2.drawContours(o_img, [points] ,-1, color=(255, 255, 255), thickness=1)
            # o_img = cv2.fillPoly(o_img, pts=[points], color=255)
            output_path = path.join(output_dir, f"{names[idx]}.png")
        cv2.imwrite(output_path, o_img)
        print(f"Created mask in {output_path}")
    # return

if __name__ == "__main__":
    gen_mask(
        images_dir='/home/dung/Seg_tel/LDC/sem_data/SEM_img',
        label_dir='/home/dung/dataset/SEM_TEL/SEM_100/SEM_label',
        output_dir='/home/dung/Project/LDC/sem_data/SEM_mask'
    )