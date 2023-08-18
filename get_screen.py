from PIL import ImageGrab
import mouseinfo
import cv2
import numpy as np

x,y=mouseinfo.absolute_position()
# 使用Pillow进行截图
bbox=(2560,0,4000,2560)
pil_image = ImageGrab.grab(bbox)
pil_image.save('screenshot.png') # 保存截图

# 将Pillow图像转换为numpy数组，并将RGB转换为BGR
# OpenCV默认使用BGR格式，而PIL使用RGB格式
cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 现在可以使用cv2处理图像了
# cv2.imshow('image', cv_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()