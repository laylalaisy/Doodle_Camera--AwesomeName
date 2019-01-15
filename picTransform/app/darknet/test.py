
# import os
# os.chdir('/Users/xierong/Desktop/Master_2016/googleMLCamp2019/objectDetection/darknet')

import python.darknet as dn

def detection(image, net, meta):
    res = dn.detect(net, meta, str.encode(image))
    return res

if __name__ == "__main__":
    dn.set_gpu(0)
    net = dn.load_net(str.encode("cfg/yolov3.cfg"),
                  str.encode("yolov3.weights"), 0)
    meta = dn.load_meta(str.encode("cfg/coco.data"))
    image = "data/googleCat.jpg"
    res = detection(image, net, meta)
    print (res)
    # for r in res:
    #     r = list(r)
    #     r[0] = str(r[0], encoding = "utf-8")
    
