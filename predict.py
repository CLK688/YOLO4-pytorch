#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import os
import cv2

yolo = YOLO()

file_path = "img"


while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        #获取图像和框位置
        r_image ,box_text = yolo.detect_image(image)
        #输出预测的框
        for value in box_text.values():
            print(value)
        r_image.show()
            
