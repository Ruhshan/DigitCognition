import cv2
import pickle
from PIL import Image
import numpy as np

def write_characters(img):
    print("writiting")
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
    #dilated = cv2.dilate(thresh,kernel,iterations = 1)
    dilated = cv2.dilate(src=thresh, kernel=kernel, anchor=(-1, -1), iterations=2)
    _, contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    
    padding = 2
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        area = w*h
        print(area)
        if area>50 and area < 2000:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            #cv2.imwrite(str(i)+".jpg",image[y:y+h,x:x+w])
            single = gray[y-padding:y+h+padding,x-padding:x+w+padding]
            ret,thresh1 = cv2.threshold(single,127,255,cv2.THRESH_BINARY)
            
            cv2.imwrite("data/crp"+str(i)+".png", thresh1)
            i=i+1
    cv2.imwrite("rect.png", image)


def sort_contours(cntrs):
    valid_contours ={}
    for cntr in cntrs:
        [x,y,w,h]=cv2.boundingRect(cntr)
        area = w*h
        if area>100 and area < 2000:
            print(x,y,w,h)
            valid_contours[(x,y)] = cntr
    sorted_contours = []

    for sk in sorted(valid_contours.keys()):
        sorted_contours.append(valid_contours[sk])
    return sorted_contours

def detect(img):
    img = cv2.resize(img,(360, 60))
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
    #dilated = cv2.dilate(thresh,kernel,iterations = 1)
    dilated = cv2.dilate(src=thresh, kernel=kernel, anchor=(-1, -1), iterations=2)
    _, contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    model_file_path = 'data/model.sav'
    loaded_model = pickle.load(open(model_file_path, 'rb'))

    base_x = Image.open("data/base.png")
    padding = 2
    contours = sort_contours(contours)
    res = ""
    ct = 0
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        area = w*h
        #print(area)
        if area>100 and area < 2000:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            single = gray[y-padding:y+h+padding,x-padding:x+w+padding]
            ret,thresh1 = cv2.threshold(single,127,255,cv2.THRESH_BINARY)
            imarr = Image.fromarray(thresh1)
            base = base_x.copy()
            base.paste(imarr)
            #base.save("rec_{}.png".format(ct), "PNG")
            ct+=1
            base = np.array(base)
            grayed = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            gf = grayed.flatten() 
            p=loaded_model.predict([gf])
            res+=str(p[0])
    cv2.imwrite("rectx.png", image)

    return res


