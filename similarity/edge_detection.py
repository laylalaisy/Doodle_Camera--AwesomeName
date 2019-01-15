import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('dataset_path/cat/0.png',0)
edges = cv2.Canny(img,100,200) #参数:图片，minval，maxval,kernel = 3

plt.subplot(121) #121表示行数，列数，图片的序号即共一行两列，第一张图
plt.imshow(img,cmap='gray') #cmap :colormap 设置颜色
plt.title('original image'),plt.xticks([]),plt.yticks([]) #坐标轴起点，终点的值
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('edge image'),plt.xticks([]),plt.yticks([])

plt.show()


