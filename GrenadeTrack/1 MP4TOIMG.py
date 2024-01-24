import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture("G:/TrainData/Grenade_BIGC/test_far.mp4")  # 获取视频对象
isOpened = cap.isOpened  # 判断是否打开
# 视频信息获取
fps = cap.get(cv2.CAP_PROP_FPS)

imageNum = 0
sum = 0
timef = 2  # 隔15帧保存一张图片

while (isOpened):

    sum += 1

    (frameState, frame) = cap.read()  # 记录每帧及获取状态

    if frameState == True and (sum % timef == 0):

        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        frame = np.array(frame)

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        imageNum = imageNum + 2
        fileName = 'G:/TrainData/Grenade_BIGC/pre/img_2/img_03' + str(imageNum) + '.jpg'  # 存储路径
        cv2.imwrite(fileName, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(fileName + " successfully write in")  # 输出存储状态

    elif frameState == False:
        break

print('finish!')
cap.release()
