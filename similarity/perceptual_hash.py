import sys
from PIL import Image
import imagehash

DOODLENUM = 1000

if __name__ == "__main__":
    label = sys.argv[1]
    photo_filename = sys.argv[2]
    doodle_foldername = sys.argv[3]
    photo_hash = imagehash.average_hash(Image.open(photo_filename))
    min_dist = float('inf')
    for i in range(DOODLENUM):
        doodle_hash = imagehash.average_hash(Image.open(doodle_foldername+str(i)+'.png'))
        dist = photo_hash - doodle_hash
        if min_dist > dist:
            min_dist = dist
            min_idx = i
    print(min_dist)
    print(min_idx)

