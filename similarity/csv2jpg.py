from PIL import Image, ImageDraw
import csv

out = Image.new("L",(49,87))
dout = ImageDraw.Draw(out)
import csv
with open('pic.txt', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        dout.point((int(row[0]) / 10,int(row[1]) / 10),fill=int(int(row[2]) * 2.55))
        #print(row[0] + " " + row[1] + " " + row[2])
out.show()
