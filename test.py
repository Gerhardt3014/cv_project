# import cv2
# def slope(image_path=None, img_frame=None):
#     if img_frame == None and image_path != None:
#         img = cv2.imread(image_path)
#     elif img_frame != None and image_path == None:
#         img = img_frame
#     else:
#         print("输入有误，检查输入")
#         return
#
# if __name__ == '__main__':
#     slope(img_frame=1)

"""
绘制不同形状的感兴趣区域
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
# image_path = "xie.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#
# # Convert from BGR to RGB
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Create a mask
# mask = np.zeros_like(image)
#
# # Define the circle's center and radius
# circle_center = (150, 150)
# circle_radius = 100
#
# # Define the trapezoid's vertices
# trapezoid_vertices = np.array([[100, 300], [200, 300], [250, 400], [50, 400]])
#
# # Draw a white circle on the mask
# cv2.circle(mask, circle_center, circle_radius, (255, 255, 255), -1)
#
# # Draw a white trapezoid on the mask
# cv2.fillPoly(mask, [trapezoid_vertices], (255, 255, 255))
#
# # Extract the ROIs
# # circle_roi = cv2.bitwise_and(image, mask)
# trapezoid_roi = cv2.bitwise_and(image, mask)
#
# # Display the original image, circle ROI, and trapezoid ROI
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
# ax[0].imshow(image)
# ax[0].set_title("Original Image")
# ax[0].axis("off")
#
# ax[1].imshow(mask)
# ax[1].set_title("Circle ROI")
# ax[1].axis("off")
#
# ax[2].imshow(trapezoid_roi)
# ax[2].set_title("Trapezoid ROI")
# ax[2].axis("off")
#
# plt.show()
# print("hello")
from functools import reduce
numbers = [1, 2, 3, 4, 5]
total_sum = reduce(lambda x, y: x + y, numbers)
