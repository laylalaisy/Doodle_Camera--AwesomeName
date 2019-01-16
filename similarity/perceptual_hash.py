from PIL import Image
import imagehash

hash = imagehash.average_hash(Image.open('0.jpg'))
min = 10000000
for i in range(999):
    otherhash = imagehash.average_hash(Image.open('./pencil_img/'+str(i)+'.png'))
    dist = hash - otherhash
    if min > dist:
        min = dist
        idx = i
print(min)
print(idx)

