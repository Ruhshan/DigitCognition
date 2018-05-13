import glob
import os
import shutil

# shape  = 'circle'

# for f in glob.glob('train/circle/*'):
#     print(f)

digname = "nine"

path = "/home/ruhshan/works/digitserver/data/train/{}/".format(digname)
moveto = "/home/ruhshan/works/digitserver/data/valid/{}/".format(digname)

files = os.listdir(path)
files.sort()
print(type(files))
for f in files[:20]:
    src = path+f
    dst = moveto+f
    shutil.move(src, dst)


    
