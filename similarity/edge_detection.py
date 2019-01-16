import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# input original image
	img_orig = cv2.imread('./dataset/doodles/image_orig/dog/9.png',0)

	# do edge detection
	img_edges = cv2.Canny(img_orig,100,200)

	# resize image: 256*256
	img_resize = cv2.resize(img_edges, (256,256), interpolation=cv2.INTER_CUBIC)

	# save resize image
	# cv2.imwrite('./dataset/doodles/image_edge/dog/0_edge.jpg', img_resize)

	# save
	for i in range(256):
		for j in range(256):
			if img_resize[i][j] != 0:
				img_resize[i][j] = 1
	np.savetxt('./dataset/doodles/edge_matrix/dog/9.csv', img_resize, delimiter = ',')

	# plt.imshow(img_resize,cmap = 'gray')
	# plt.show()
