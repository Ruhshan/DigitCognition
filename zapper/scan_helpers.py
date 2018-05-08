from imageprocessing.datatoimage import make_image
from imageprocessing.roi_extraction import extract_roi, extract_roi_2, extract_roi_3
from imageprocessing.detection import detect
import cv2
import numpy
import itertools
import imutils

def best_candidate(chars):
    unq = set(chars)
    counts={}
    for c in unq:
        counts[chars.count(c)]=c
    mx = max(counts.keys())
    return counts[mx]

def analyze_results(results):
    valid_results = [result for result in results if len(result)==48]
    from pprint import pprint
    pprint(valid_results)

    constructed_str=''
    if len(valid_results)==0:
        return "Zap again"
    else:
        for i in range(48):
            chars=[]
            for j in range(len(valid_results)):
                chars.append(valid_results[j][i])
            constructed_str+=best_candidate(chars)
        
        return constructed_str
def locate_t(image):
    try:
        #img = cv2.imread('rotating.png',0)
        #img.transpose(2,0,1).reshape(3,-1)
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        template = cv2.imread('t_template.png', 0)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        return top_left
    except:
        return (0,0)

def angle_correct_then_detect(roi):
    results = []
    for a in (0,90,180,270):
        rotated = imutils.rotate_bound(roi, a)
        if rotated.shape[0]>rotated.shape[1]:
            cv2.imwrite("rotate_"+str(a)+".png", rotated)
            t_location = locate_t(rotated)
            print(a, "t_location", t_location)
            #if t_location[0]>200 and t_location[1]>120:
            cv2.imwrite("rotate_"+str(a)+"_sel.png", rotated)
            try:
                detected = detect(rotated, a)
                print('detecting',a,detected)
                detected = detected.replace('-','')
                print("length ", len(detected))
                if len(detected)==48:
                    print("appended")
                    results.append(detected)
                    #sometimes correct code is returned from upside down code
                    results.append(detected[::-1])
            except:
                print(a)
    print(results)
    decoded = []
    for r in results:
        decoded.append([decode(r), decode(r).count('-')])
    min_unknown=48
    code=''
    for d in decoded:
        if d[1]< min_unknown:
            min_unknown = d[1]
            code = d[0]
    return code

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
    result=''
    if ret == 0:
        cv2.imwrite('received.png', img)
        #img = cv2.imread('received.png')
        # try:
        #     roi = extract_roi_2(img)
        #     if type(roi) == numpy.ndarray:
        #         #trying to read the code from roi
        #         print("trying to correct rotation")
        #         result = angle_correct_then_detect(roi)
        #         # detected = detect(roi)
        #         # result=detected.replace('-','')
        #     else:
        #         #return HttpResponse("Zap again")
        #         print("Unable to extract roi")
        #         result="----------------"
        # except:
        #     result = "----------------"
        # # When roi is extracted its a 2d array 
        
    return "ok"

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
