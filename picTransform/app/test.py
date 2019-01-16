import cv2
import numpy

def splitBox(image, x, y, width, height):
    img = cv2.imread("static/images/current.jpg")
    print(type(img))
    cropImg = img[round(y-0.5*height) : round(y+0.5*height), round(x-0.5*width) : round(x+0.5*width)]
    cv2.imwrite("test.jpg",cropImg)

def joinBox():
    image = "static/images/current.jpg"
    img = cv2.imread(image)
    # size = img.shape
    # height = size[0]
    # width = size[1]
    emptyImage = numpy.zeros(img.shape, numpy.uint8) + 255
    cv2.namedWindow('lena') # 创建窗口
    cv2.imshow('lena', emptyImage) # 在窗口显示图像
    cv2.waitKey(0) # 保持等待直到按下任意键
def testReturn():
    a = 1
    b = 2
    return (a, b)

if __name__ == "__main__":
    #splitBox("test", 224.18377685546875, 378.4237060546875, 178.60214233398438, 328.1665954589844)
    # img = cv2.imread("static/images/current.jpg")
    # size = img.shape
    # print(size, type(size[0]), size[1])
    # joinBox()
    # x = 231.57638549804688
    # y = 238.78427124023438
    # width = 82.76309967041016
    # height = 272.6578674316406
    # image = "static/images/0.jpg"
    # img = cv2.imread(image)
    # i = img[round(y-0.5*height) : round(y+0.5*height), round(x-0.5*width) : round(x+0.5*width)]
    # cv2.imshow('lena', i) # 在窗口显示图像
    # cv2.waitKey(0) # 保持等待直到按下任意键
    a = [1, 2, 3, 4]
    for aa in a:
        if aa == 1 :
            a.append(5)
            a.remove(aa)
    print (a)