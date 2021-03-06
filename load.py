from os.path import splitext
from os import listdir
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# imgs_dir='/home/holmes/SmartCar/img-segmenation/Dataset/Train'
# ids = [splitext(file)[0] for file in listdir(imgs_dir)if not file.startswith('.')]
# print(ids)
train_dir ='/home/holmes/SmartCar/img-segmenation/Dataset/Train'+'/'+'0000'+'/'+'label'
path=glob.glob(train_dir+'.*')[0]
# print(path)
img =Image.open(path).convert('L')
# img.show()
# npimg =np.array(img)
pixels = img.load()
for x in range(img.width):
    for y in range(img.height):
        pixels[x, y] = 255 if pixels[x, y] > 0 else 0
# print(npimg)
img.show()