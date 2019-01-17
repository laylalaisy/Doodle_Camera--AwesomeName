from app import app
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from datetime import timedelta
import numpy
import sys
from PIL import Image
import imagehash
import shutil

# import sys
# sys.path.append('/home/rongxie_sx/AwesomeName/picTransform/app/darknet/python/')
# import darknet as dn

from .darknet.python import darknet as dn
#import darknet.python.darknet as dn


def detection(image):
    basepath = os.path.dirname(__file__)
    final_path = os.path.join(basepath, 'darknet')
    dn.set_gpu(0)
    net = dn.load_net(str.encode(final_path+"/cfg/yolov3.cfg"),
                  str.encode(final_path+"/yolov3.weights"), 0)
    meta = dn.load_meta(str.encode(final_path+"/cfg/coco.data"))
    # meta = dn.load_meta(str.encode(final_path+"/cfg/combine9k.data"))

    res = dn.detect(net, meta, str.encode(basepath+"/static/images/"+image))
    return res

def splitOneBox(image, x, y, width, height, order):
    basepath = os.path.dirname(__file__)
    img = cv2.imread(image)
    cropImg = img[round(y-0.5*height) : round(y+0.5*height), round(x-0.5*width) : round(x+0.5*width)]
    cv2.imwrite(basepath+"/static/images/"+str(order)+".jpg",cropImg)

def splitBox(objectList):
    print (objectList)
    basepath = os.path.dirname(__file__)
    image = basepath + "/static/images/current.jpg"
    order = 0
    box = {}
    order = 0
    while order < len(objectList):
        obj = objectList[order]
        if type(obj[0]) == bytes:
            category = str(obj[0], encoding="utf-8")
        else:
            category = obj[0]
        confidence = obj[1]
        x = obj[2][0]
        y = obj[2][1]
        width = obj[2][2]
        height = obj[2][3]
        print(category)
        if category == 'person':
            objectList.remove(obj)
            y1 = y - 7 / 16 * height
            w1 = height / 8
            h1 = height / 8
            y2 = y - 3 / 16 * height
            w2 = width
            h2 = height / 8 * 3
            y3 = y + 0.25 * height
            w3 = width
            h3 = height / 2
            headTuple = (x, y1, w1, h1)
            bodyTuple = (x, y2, w2, h2)
            legTuple = (x, y3, w3, h3)
            head = ['face', 0.9, headTuple]
            body = ['t-shirt', 0.9, bodyTuple]
            leg = ['pants', 0.9, legTuple]
            objectList.append(head)
            objectList.append(body)
            objectList.append(leg)
            
        else:
            splitOneBox(image, x, y, width, height, order)
            box[order] = {}
            box[order]['label'] = category
            box[order]['path'] = basepath+'/static/images/'+str(order)+'.jpg'
            box[order]['x'] = x
            box[order]['y'] = y
            box[order]['width'] = width
            box[order]['height'] = height
            order += 1
    return (box, objectList)

DOODLENUM = 1000

def generateDoodle(label, photo_filename, width, height):
    basepath = os.path.dirname(__file__)
    doodle_foldername = basepath + "/ndjson/image_orig/"+label+"/"
    firstimg = doodle_foldername + "0.png"
    print("firstimg:" + firstimg)
    try:
        with open(firstimg, "r") as f:
            # resize photo
            print("##exist##")
            photo = cv2.imread(photo_filename,0)
            photo_resize = cv2.resize(photo, (256, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(photo_filename, photo_resize)

            # find similar doodle
            photo_hash = imagehash.average_hash(Image.open(photo_filename))
            min_dist = float('inf')
            for i in range(DOODLENUM):
                doodle_hash = imagehash.average_hash(Image.open(doodle_foldername+str(i)+'.png'))
                dist = photo_hash - doodle_hash
                if min_dist > dist:
                    min_dist = dist
                    min_idx = i
            print('output_doodle:' + doodle_foldername+str(min_idx)+'.png')

            # input original image
            img_orig = cv2.imread(doodle_foldername+str(min_idx)+'.png',0)

            # resize image: 256*256
            img_resize = cv2.resize(img_orig, (round(width),round(height)), interpolation=cv2.INTER_CUBIC)

            # save resize image
            cv2.imwrite(photo_filename, img_resize)

            # plt.imshow(img_resize,cmap = 'gray')
            # plt.show()
    except Exception as e:
        print("##Not Exist##")
        os.remove(photo_filename)
        return

    
def joinBox(objectList):
    basepath = os.path.dirname(__file__)
    image = basepath + "/static/images/current.jpg"
    img = cv2.imread(image)
    emptyImg = numpy.zeros(img.shape, numpy.uint8) + 255
    order = 0
    for obj in objectList:
        x = obj[2][0]
        y = obj[2][1]
        width = obj[2][2]
        height = obj[2][3]
        # if not os.path.exist(basepath + "/static/images/" + str(order) + ".jpg"):
        #     order+=1
        #     continue
        subImg = cv2.imread(basepath + "/static/images/" + str(order) + ".jpg")
        if subImg is None:
            order+=1
            continue
        yUp = round(y-0.5*height)
        yBottom = round(y+0.5*height)
        xLeft = round(x-0.5*width)
        xRight = round(x+0.5*width)
        try:
            emptyImg[yUp : yBottom, xLeft : xRight] = cv2.resize(subImg, (xRight-xLeft, yBottom-yUp))
        except Exception as e:
            print(str(obj[0]))
            order += 1
            continue
        cv2.imwrite(basepath+"/static/images/join.jpg", emptyImg)
        order += 1

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/')
@app.route('/index', methods = ['GET', 'POST'])

def index():
    if request.method == 'POST':
        basepath = os.path.dirname(__file__)

        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "Please check picture form (png, PNG, jpg, JPG and bmp are allowed.)"})

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'current.jpg'), img)
        image = 'current.jpg'
        res = detection(image)
        (boxes, objectlist) = splitBox(res)
        for key, value in boxes.items():
            label = value['label']
            photo_filename = value['path']
            width = value['width']
            height = value['height']
            generateDoodle(label, photo_filename, width, height)   
        joinBox(objectlist)
    
        return render_template('index_ok.html')
 
    return render_template('index.html')
