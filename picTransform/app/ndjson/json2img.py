import sys
import json
from scipy import interpolate
import pylab as pl

# python json2img.py dog ./dataset/doodles/json/full_simplified_dog.json ./dataset/doodles/image_orig/dog/

if __name__ == "__main__":
    label = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    print("hello")
    f = open(input_file)    # ("./dataset/doodles/json/full_simplified_dog.json")
    setting = json.load(f)
    for j in range(0,1000):
        for i in range(0,len(setting[j]['drawing'])):
            x = setting[j]['drawing'][i][0]
            y = setting[j]['drawing'][i][1]
            f=interpolate.interp1d(x,y,kind="slinear")
            pl.plot(x,y,'k')
        ax = pl.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        pl.axis('off')
        pl.savefig(output_file+"%d.png"%j)
        pl.close()
