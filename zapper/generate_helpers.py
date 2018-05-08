import itertools
import cv2
img = cv2.imread('imageprocessing/zap_16_2.png', 0)
import random
from PIL import Image
# t = cv2.imread('shapes/minus.jpg', cv2.IMREAD_UNCHANGED)
# c = cv2.imread('shapes/o.png', cv2.IMREAD_UNCHANGED)
# s = cv2.imread('shapes/cross.png', cv2.IMREAD_UNCHANGED) 

t = cv2.imread('shapes/trianglef.png', cv2.IMREAD_UNCHANGED)
c = cv2.imread('shapes/circle.png', cv2.IMREAD_UNCHANGED)
s = cv2.imread('shapes/sq_f.png', cv2.IMREAD_UNCHANGED) 

 

t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)


t = cv2.resize(t, (50, 50))
c = cv2.resize(c, (50, 50))
s = cv2.resize(s, (50, 50))

shapes ={'t':t, 'c':c, 's':s}

def get_combinations_dict():
    with open('combinations.txt') as f:
        combinations={l.split(';')[0]:l.split(';')[1] for l in f.read().split('\n')}
        return combinations
def generate_zap(n):
    #combinations=list(map("".join, itertools.combinations_with_replacement('cst',3)))
    #combinations=list(map("".join, itertools.combinations_with_replacement('ttt',3)))
    combinations = get_combinations_dict()
    #print(combinations)
    symbols=[]
    for c in n:
        symbols.extend(list(combinations[c]))

    symbols=symbols[::-1]
    for j in range(0, 4):
            y_offset = 40 + (j*61)
            for i in range(6):
                x_offset = 45+ (i*80)
                img[y_offset:t.shape[0]+y_offset, x_offset:t.shape[1]+x_offset] = shapes[symbols.pop()]
                #img[y_offset:t.shape[0]+y_offset, x_offset:t.shape[1]+x_offset] = shapes['s']

    for j in range(0, 4):
        y_offset = 395 + (j*61)
        for i in range(6):
            x_offset = 45+ (i*80)
            img[y_offset:t.shape[0]+y_offset, x_offset:t.shape[1]+x_offset] = shapes[symbols.pop()]
            #img[y_offset:t.shape[0]+y_offset, x_offset:t.shape[1]+x_offset] = shapes['s']
    
    logo = Image.open("imageprocessing/connect_logo2.png")
    ratio=0.15
    logo = logo.resize( [int(ratio * s) for s in logo.size] )
    # logo = cv2.resize(logo, (0,0), fx=0.1, fy=0.1)
    
    img_c = Image.fromarray(img)
    img_c.paste(logo, (60,285), logo)
    return img_c
    #return Image.fromarray(resized)

