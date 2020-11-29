#-------------------------------------#
#       对多张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import os
import cv2
import numpy as np
yolo = YOLO()

file_path = "img"
for file in os.listdir(file_path):
    filename = os.path.join(file_path,file)
    try:
        image = Image.open(filename)
    except:
        print('Open Error! Try again!')
        continue
    else:
        #获取图像和框位置
        r_image ,box_text = yolo.detect_image(image)
        #输出预测的框
        for value in box_text.values():
            print(value)
        frame = np.array(r_image)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imwrite("result/"+file, frame)
        # r_image.show()
            