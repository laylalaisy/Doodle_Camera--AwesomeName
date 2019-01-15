import json
from scipy import interpolate
import pylab as pl
f = open("./dataset/doodles/json/dog/full_simplified_dog.json")
setting = json.load(f)
for j in range(0,10):
    for i in range(0,len(setting[j]['drawing'])):
        x = setting[j]['drawing'][i][0]
        y = setting[j]['drawing'][i][1]
        f=interpolate.interp1d(x,y,kind="slinear")
        pl.plot(x,y,'k')
    ax = pl.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    pl.axis('off')
    pl.savefig("./dataset/doodles/image/dog/%d.png"%j)
    pl.close()
