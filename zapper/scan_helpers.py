from imageprocessing.datatoimage import make_image
from imageprocessing.roi_extraction import extract_roi, extract_roi_2, extract_roi_3
import cv2
import numpy
import itertools
import imutils

from imageprocessing.detection import write_characters, detect


def process_payload(payload):
    """
    Entry point of processing the received payload. Makes image from base64 string, extracts roi, tries to detect the code or 
    prompts to zap again

    Parameters
    ----------
    payload:    String
                base64 encoded image

    Returns
    ----------
    result:     String
                zap code as character string eg:CTSTCTCTC...CSSTT or 'Zap again' 
    """

    # Convertion of payload string to image array for opencv
    ret, img = make_image(payload)#ret is 0 when conversion is successful or 1 when not
    result='Unable to detect'
    if ret == 0:
        cv2.imwrite('received.png', img)
        #img = cv2.imread('received.png')
        try:
            roi = extract_roi_2(img)
            result = detect(roi) 
            #cv2.imwrite("roi.png", roi)
            #write_characters(roi)

        except:
            result = "----------------"
        # # When roi is extracted its a 2d array 
        
    return result

def get_combinations_map():
    with open('recognition.txt') as f:
        combinations={l.split(';')[0]:l.split(';')[1] for l in f.read().split('\n')}
        return combinations
def error_check(s):
    if '-' in s:
        return '!'
    else:
        return '#'
def decode(a):
    ct=0
    combination_map=get_combinations_map()
    # combinations=list(map("".join, itertools.combinations_with_replacement('cst',3)))

    # for c in combinations:
    #     combination_map[c]=ct
    #     ct+=1
    decoded=""
    
    for i in range(0,48,3):
        #print(combination_map[a[i:i+3]])
        sub = a[i:i+3]
        try:
            decoded+=str(combination_map[sub])
        except:
            #print(sub, '---')
            decoded+='---'
    #print('#########')
    return decoded


def process_payload_get_roi(payload):
    ret, img = make_image(payload)
    cv2.imwrite('received.png', img)
    roi = extract_roi_3(img)
    extract_roi_2(img)
    # center , wh, theta = roi 
    # x,y = center
    # w,h = wh   
    # roi_metrices = {'x':x, 'y':y,'w':w,'h':h, 'theta':theta} 
    return roi
