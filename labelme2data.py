import labelme
import os, sys
path="/home/dung/Seg_tel/LDC/sem_data/SEM_label/"

i=0
converted_dir = '/home/dung/Seg_tel/LDC/sem_data/SEM_label/converted/'

for item in os.listdir(path):
   
   if item.endswith(".json"):
      print(item)
      if os.path.isfile(path+item):
         my_dest =converted_dir + str(i)
         file = path + item
         print(my_dest)
         os.system("labelme_json_to_dataset "+ file +" -o "+converted_dir)
         i=i+1
