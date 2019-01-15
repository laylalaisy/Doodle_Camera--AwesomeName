from app import app
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from datetime import timedelta
from .darknet.python import darknet as dn

def detection(image):
    basepath = os.path.dirname(__file__)
    final_path = os.path.join(basepath, 'darknet')
    dn.set_gpu(0)
    net = dn.load_net(str.encode(final_path+"/cfg/yolov3.cfg"),
                  str.encode(final_path+"/yolov3.weights"), 0)
    meta = dn.load_meta(str.encode(final_path+"/cfg/coco.data"))
    res = dn.detect(net, meta, str.encode(basepath+"/static/images/"+image))
    return res

def splitBox(image, x, y, width, height):
    # basepath = os.path.dirname(__file__)
    img = cv2.imread(image)
    cropImg = img[round(y-0.5*height) : round(y+0.5*height), round(x-0.5*width) : round(x+0.5*width)]
    #cv2.imwrite("test.jpg",cropImg)
    return cropImg

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/')
@app.route('/index', methods = ['GET', 'POST'])

def index():
    if request.method == 'POST':
        
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "Please check picture form (png, PNG, jpg, JPG and bmp are allowed.)"})

        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))
        f.save(upload_path)
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'current.jpg'), img)
        res = detection('current.jpg')
        return render_template('index_ok.html', res = res)
 
    return render_template('index.html')


