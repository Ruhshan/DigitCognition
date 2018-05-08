import cv2
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from collections import OrderedDict
from pandas import Series

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *



arch=resnet34
PATH = "imageprocessing/dummyshapes"
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, 1))
learn = ConvLearner.pretrained(arch, data, precompute=False)
#learn.fit(0.01, 4)
#learn.load('resnet34_connect')
learn.load('resnet34_rlearned')
def resnet_pred(target):
    #print(target.shape)
    global arch, learn, ct
    trn_tfms, val_tfms = tfms_from_model(arch,100)
    color_target =cv2.cvtColor(target, cv2.COLOR_GRAY2RGB).astype(np.float32)/255
    # cv2.imwrite("target_print/"+str(ct)+".png", target)
    im = val_tfms(color_target)
    try:
        preds = learn.predict_dl(im[None])
    except:
        preds = learn.predict_array(im[None])
    val_pred = np.argmax(preds)
    return val_pred+1



def get_key_points(blurred, im, kp_area, a):
    

    params = cv2.SimpleBlobDetector_Params()


    # params.filterByArea = True
    # params.minArea = kp_area

    params.minThreshold = 10
    params.maxThreshold = 200

    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.maxConvexity = 1

    detector = cv2.SimpleBlobDetector_create(params)

    ##only croped or croped and resized
    keypoints = detector.detect(blurred)

    ## croped and thresholded
    #keypoints = detector.detect(im)
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite(str(a)+"_im_wth_kp.png", im_with_keypoints)

    # dictionary where keypoints are stored against its x,y co-ordinate
    key_points_dict = {}
    for keypoint in keypoints:
        x, y = keypoint.pt[0], keypoint.pt[1]
        key_points_dict[(x,y)] = keypoint

    # Keypoints are ordered
    ordered_keypoints = OrderedDict(sorted(key_points_dict.items(), key=lambda t: t[0]))
    return ordered_keypoints

def detect(imc, a):
    """
    Tries separate each shapes as blob in the image and detects its corresponding label

    Parameters
    ----------
    imc:    ndarray
    
    Returns
    ---------
    vals:   String
            A string of corresponding label for each shape. Eg: 'ccttct-stscct...'
    """
    #imc = cv2.imread('to_detect.png', cv2.IMREAD_COLOR)
    
    try:
        imc = cv2.resize(imc, (240, 260))
    except:
        return "Zap Again x"
    cv2.imwrite('to_detect.png', imc)
    #print('imc shape', imc.shape)
    im = cv2.cvtColor(imc, cv2.COLOR_BGR2GRAY)
    im2 = im.copy()
    ct=384
    blurred = cv2.GaussianBlur(im, (5, 5), 2)
    cv2.imwrite('blurred.png', blurred)
    kp_len=0
    kp_area=130
    while 1:
        ordered_keypoints = get_key_points(blurred,im, kp_area,a)
        kp_area-=1
        kp_len = len(ordered_keypoints)
        print("lllll", kp_len)
        if kp_len >= 48:
            break
        if kp_area < 60:
            break
    # a trained model for shape detection
    model_file_path = 'imageprocessing/model_big_aug.sav'
    loaded_model = pickle.load(open(model_file_path, 'rb'))

    # this dictionary will contain shape labels stored agains shapes co-ordinates
    shape_dict = {}
    min_x,min_y = 100,100
    det = cv2.imread("imageprocessing/detected_base.png")
    for x, y in ordered_keypoints.keys():
        ct+=1
        # From
        if min_x<x:
            min_x = x
        if min_y<y:
            min_y = x
        if y<14:
            img_crop = im2[int(y)-int(y):int(y)+14,
                           int(x)-14:int(x)+14]
        else:
            img_crop = im2[int(y)-14:int(y)+14,
                           int(x)-14:int(x)+14]
        r, imt = cv2.threshold(img_crop,200,255, cv2.THRESH_OTSU)
        height, width = blurred.shape[:2]
        res = cv2.resize(imt, (100, 100), interpolation=cv2.INTER_AREA)
        

        #cv2.imwrite("new_classification_data2/star/star_"+str(ct)+".png", res)
        res_pred = resnet_pred(res)
        X = res.flatten()
        #p=loaded_model.predict([X])
        p=[res_pred]
        font = cv2.FONT_HERSHEY_SIMPLEX
        #print("gaus ", p, "resnet", res_pred)

        if p[0]==1:
            shape_dict[(x,y)]='c'
            cv2.putText(imc, 'c', (int(x)-3, int(y)-3), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(det, 'c', (int(x), int(y)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if p[0] == 2:
            shape_dict[(x, y)] = 's'
            cv2.putText(imc, 's', (int(x) - 3, int(y) - 3), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(det, 's', (int(x), int(y)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if p[0] == 3:
            shape_dict[(x, y)] = 't'
            cv2.putText(imc, 't', (int(x) - 3, int(y) - 3), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(det, 't', (int(x), int(y) ), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
    
    xs = []
    ys = []

    for s in list(shape_dict.keys()):
        xs.append(s[0])
        ys.append(s[1])
    
    o=open(str(a)+"_vals", "w")
    o.write(";".join(list(map(str,ys))))
    o.close()
    
    lines = []
    line = []
    ct = 0
    s= Series(ys)
    #q = s.quantile([0.16, 0.32, 0.48, 0.64, 0.80, 0.96])
    q = s.quantile([0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1])
    
    #print(q)
    clusters = {}
    dists=[]
    for j in q.index:
        clusters[j]=[]
        for i in shape_dict.keys():
            d = abs(q[j]-i[1])
            dists.append(d)
            if d < 7:
                clusters[j].append(i)
    
    
    #print("ddddd", sorted(dists))
    #for cluster in clusters.values():
        #print(cluster)
        # for c in cluster:
        #     vals2+=shape_dict[c]
        # vals2+="-"
    #
    #print("vals22222", vals2)
    for y in sorted(ys):
        for s in shape_dict.keys():
            if s[1] == y:
                line.append(s)
                ct += 1
                if ct % 6 == 0:
                    lines.append(line)
                    line = []
    vals=""
    for line in lines:
        for l in sorted(line):
            vals+=shape_dict[l]
        vals+="-"
    ##experimental
    line=[]
    lines=[]
    lc=0
    stop = 0
    ct=0
    for y in sorted(ys):
        if stop==1:
            break
        for s in shape_dict.keys():
            if s[1] == y:
                line.append(s)
                ct += 1
                if ct % 6 == 0:
                    lines.append(line)
                    line = []
                    lc+=1
                    if lc==4:
                        stop=1
                    
    vals2=""
    for line in lines:
        for l in sorted(line):
            vals2+=shape_dict[l]
        vals2+='-'
    
    line=[]
    lines=[]
    lc=0
    stop = 0
    ct=0
    for y in sorted(ys)[::-1]:
        if stop==1:
            break
        for s in shape_dict.keys():
            if s[1] == y:
                line.append(s)
                ct += 1
                if ct % 6 == 0:
                    lines.append(line)
                    line = []
                    lc+=1
                    if lc==4:
                        stop=1
                    
    for line in lines[::-1]:
        for l in sorted(line):
            vals2+=shape_dict[l]
        vals2+='-'
    print("vals2s",vals2)
    #         vals2+=shape_dict[l]
    #     vals2+="-"
    # print("valsss22", vals2)
    ##experimental
    cv2.imwrite(str(a)+'_detected.png', imc)
    cv2.imwrite(str(a)+'_detected_text.png', det)
    import pprint
    print("valsss", vals)
    return vals2


#cv2.imshow('detected', imc)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
