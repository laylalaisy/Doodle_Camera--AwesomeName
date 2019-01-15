import cv2
import numpy as np
import matplotlib.pyplot as plt

# input original image
img_ori = cv2.imread('./dataset/photos/image/dog/0.jpg',0)

# do edge detection
img_edges = cv2.Canny(img_ori,100,200)

# resize image: 256*256
img_resize = cv2.resize(img_edges, (256,256), interpolation=cv2.INTER_CUBIC)

# save 
cv2.imwrite('./dataset/photos/image/dog/0_edge.jpg', img_resize)

print(img_resize)

plt.imshow(img_resize,cmap = 'gray')
plt.show()
