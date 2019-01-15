import cv2

def splitBox(image, x, y, width, height):
    img = cv2.imread("darknet/data/dog.jpg")
    cropImg = img[round(y-0.5*height) : round(y+0.5*height), round(x-0.5*width) : round(x+0.5*width)]
    cv2.imwrite("test.jpg",cropImg)

if __name__ == "__main__":
    splitBox("test", 224.18377685546875, 378.4237060546875, 178.60214233398438, 328.1665954589844)

