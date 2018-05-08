#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:27:48 2017

@author: ruhshan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:30:17 2017

@author: ruhshan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 09:45:46 2017

@author: ruhshan
"""

from imageprocessing.pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2
import numpy as np
from matplotlib import pyplot as plt
from .convenience_plotting_functions import dd, ss
from .contours_functions import angle_cos, rank, mask_image

area=1
template =  cv2.imread('imageprocessing/hor_connect.png',0)
tw, th = template.shape[::-1]
# logo = cv2.imread('connect_logo.png', -1)
# logo = cv2.resize(logo, (250, 60))
def crop_minAreaRect(img, rect):
    """
    Crops a given rectangular portion from image and tries to fix angular deviation

    Parameters
    -----------
    img:    ndarray
            Image array
    rect:   ndarray
            rectangle generated from minAreaRect function

    Returns
    ----------
    img_cropped:    ndarray
                    croped and rotated roi
    """
    angle = rect[2]

    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rotated = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    

    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_cropped = img_rotated[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_cropped
def crop_minAreaRect2(img, rect):
    """
    Crops a given rectangular portion from image and tries to fix angular deviation

    Parameters
    -----------
    img:    ndarray
            Image array
    rect:   ndarray
            rectangle generated from minAreaRect function

    Returns
    ----------
    img_cropped:    ndarray
                    croped and rotated roi
    """
    angle = rect[2]

    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rotated = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    

    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    return pts
def get_perspective_transformed(img, rect):
    box = cv2.boxPoints(rect)
    pts1 = np.float32([box[1], box[2], box[0], box[3]])

    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (300, 300))
    return dst

def fix_orientation(img):
    """
    Based on the location of 'connect logo' it rotates the image or returns the image as it receives

    Parameters
    ----------
    img:    ndarray
            and opencv image array 
    """
    h, w,c = img.shape

    loc=match_loc(img)
    print("loc is here",loc, "h, w:", h, w)

    # if connect logo is positioned in vertical middle then image doesn't need rotation
    if loc[1]>160 and loc[1]<240:
        #print("o")      
        return img
    else:
        rotated = imutils.rotate(img, 90)
        
        return rotated


def match_loc(img):
    """
    Returns the location of connect logo in received image. Logo is detected using template matching
    """
    h,w,c = img.shape
    rth = cv2.resize(template, (w, th))

    gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gr,rth,cv2.TM_CCOEFF)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    
    return top_left

def rm_logo(img):
    h, w,c = img.shape

    loc=match_loc(img)

    if loc[1]>80 and loc[1]<120:
        #print("o")
        imgu = img[:round(h*0.4)+2,]
        imgd = img[round(h*0.6)+8:,]
        #cv2.imshow('rmu', imgu)
        #cv2.imshow('rmd', imgd)

        #print("res",res)
        connectless = np.concatenate((imgu, imgd))

        return connectless
    else:
        rotated = imutils.rotate(img, 90)
        imgu = rotated[:round(h*0.4)+2,]
        imgd = rotated[round(h*0.6)+8:,]
        connectless = np.concatenate((imgu, imgd))

        return connectless

def extract_roi(image):
    """
    From an image detects the biggest square then crops it and returns the croped portion as roi

    Parameters
    ----------
    image:  ndarray
            opencv compatible image
    
    returns:
            ndarray
            biggest square in the image croped, angulare deviation fixed and rotated if necessary
            
    """
    
    image = cv2.imread('received.png',0)
    cv2.imshow('rec', image)#####
    image = cv2.resize(image, (600,600))
    ratio = image.shape[0] / float(image.shape[0])
    cv2.imshow('res', image)###
    ## only croped or croped and resized
    # Edge detection using canny
    edge = cv2.Canny(image, 100, 200)
    # Contour detection
    cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    ## croped and edged
    # new_thresh=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # #edge = cv2.Canny(new_thresh, 0, 2)
    # cv2.imwrite('edged.png', new_thresh)
    # cnts = cv2.findContours(new_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    ## croped and thresholded
    #new_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cnts = cv2.findContours(new_thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #return image

    # proper contour array based on cv version
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cv2.drawContours(image, cnts, -1, (0,255,60), 3)
    cv2.imshow('iii', image)
    #instance of shape detector class
    sd = ShapeDetector()

    # a list that will contain shape name as 'c' for circle, 's' for square for corresponding contour
    shape_array=[] 
    # a list that will contain perimeter of contour and and that contour together 
    squares = []
    for c in cnts:
             # compute the center of the contour, then detect the name of the
             # shape using only the contour
        M = cv2.moments(c)

        try:
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
        except:
            cX = 400
            cY = 500

        shape, peri = sd.detect(c)
        shape_array.append(shape)

        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        print(shape, peri)
        if shape=="s":
            squares.append([peri, c])
    # this is contain countour of square having maximum perimeter
    max_square = [0, []]

    for s in squares:

        if s[0]>max_square[0]:
            max_square = s
    
    
    try:
        # cal culating minimum surrounding rectangle
        rect = cv2.minAreaRect(max_square[1])
        #rect = cv2.minAreaRect(new_thresh)
        w, h = rect[1]
        n_area = w * h
        print(n_area)
        # crops when area of minimum surrounding rectangle is between a range
        if n_area > 80000 and n_area < 300000:
            # croping minimum surrounding rectangle from image.  
            img_croped = crop_minAreaRect(image, rect)
            
            # Though angular deviation is fixed but sometimes image's orientation becomes 90 degree off to the right. Thus try to fix orientation
            #img_croped = fix_orientation(img_croped)
            #print("CRP")
            #cv2.imwrite('croped', img_croped)
            #roi = rm_logo(img_croped)
            cv2.imwrite('image_croped.png', img_croped)
            return img_croped
            
    except:
        pass

    #
    # cv2.drawContours(image, max_square[1], -1, (0, 255, 0), 5)
    #
    #
    #
    #
    #
    # cv2.imshow('frame', image)
    # exec(open('detection.py').read())
    # #exec(open('make_table.py').read())
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def extract_roi_2(img):
    #img = cv2.imread('received.png',)
    img = cv2.resize(img, (600,480))
    img_copy = img.copy()[:,:,::-1] # color channel plotting mess http://stackoverflow.com/a/15074748/2256243
    height = img.shape[0]
    width = img.shape[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # matrix of ones

    squares = []
    all_contours = []

    for gray in cv2.split(img):
        dilated = cv2.dilate(src = gray, kernel = kernel, anchor = (-1,-1))

        blured = cv2.medianBlur(dilated, 7)

        # Shrinking followed by expanding can be used for removing isolated noise pixels
        # another way to think of it is "enlarging the background"
        # http://www.cs.umb.edu/~marc/cs675/cvs09-12.pdf
        small = cv2.pyrDown(blured, dstsize = (int(width / 2), int(height / 2)))
        oversized = cv2.pyrUp(small, dstsize = (width, height))

        # after seeing utility of later thresholds (non 0 threshold results)
        # try instead to loop through and change thresholds in the canny filter
        # also might be interesting to store the contours in different arrays for display to color them according
        # to the channel that they came from
        for thrs in range(0, 255, 26):
            if thrs == 0:
                edges = cv2.Canny(oversized, threshold1 = 0, threshold2 = 50, apertureSize = 3)
                next = cv2.dilate(src = edges, kernel = kernel, anchor = (-1,-1))
            else:
                retval, next = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)

            contours = cv2.findContours(next, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if imutils.is_cv2() else contours[1]
            # how are the contours sorted? outwards to inwards? would be interesting to do a PVE
            # sort of thing where the contours within a contour (and maybe see an elbow plot of some sort)
            for cnt in contours:
                all_contours.append(cnt)
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)

    sorted_squares = sorted(squares, key=lambda square: rank(square, img))

    # if len(sorted_squares) and rank(sorted_squares[0], img) < 3:
    #     cv2.drawContours(img, squares, -1, (0,255,255), 1) # draw all found squares
    #     cv2.drawContours(img, [sorted_squares[0]], -1, (0,255,60), 3)
    #     awef = cv2.imwrite('./out/' + filename, img)
    #     if awef:
    #         print("yes")
    #     with open('logs/square_found.csv', 'a') as csvfile:
    #         writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    #         writer.writerow([filename, json.dumps(sorted_squares[0].tolist()), height, width])
    # else:
    #     with open("logs/squareless.txt", "a") as f:
    #         f.write(filename + "\n")

    ### Some plotting code
    # plt.subplot2grid((2,5), (0,0)),plt.imshow(img_copy)
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # gray = cv2.split(img_copy)[0]
    # plt.subplot2grid((2,5), (0,1)),plt.imshow(gray, cmap = 'gray')
    # plt.title('Single Channel'), plt.xticks([]), plt.yticks([])
    # # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # dilated = cv2.dilate(src = gray, kernel = kernel, anchor = (-1,-1))
    # plt.subplot2grid((2,5), (0,2)),plt.imshow(dilated, cmap = 'gray')
    # plt.title('Dilated'), plt.xticks([]), plt.yticks([])
    # blured = cv2.medianBlur(dilated, 7)
    # plt.subplot2grid((2,5), (0,3)),plt.imshow(blured, cmap = 'gray')
    # plt.title('Median Filter'), plt.xticks([]), plt.yticks([])

    # # Shrinking followed by expanding can be used for removing isolated noise pixels
    # # another way to think of it is "enlarging the background"
    # # http://www.cs.umb.edu/~marc/cs675/cvs09-12.pdf
    # small = cv2.pyrDown(blured, dstsize = (int(width / 2), int(height / 2)))
    # oversized = cv2.pyrUp(small, dstsize = (width, height))
    # plt.subplot2grid((2,5), (0,4)),plt.imshow(oversized, cmap = 'gray')
    # plt.title('Resized'), plt.xticks([]), plt.yticks([])

    # edges = cv2.Canny(oversized, threshold1 = 0, threshold2 = 50, apertureSize = 3)
    # plt.subplot2grid((2,5), (1,0)),plt.imshow(edges, cmap = 'gray')
    # plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
    # dilated = cv2.dilate(src = edges, kernel = kernel, anchor = (-1,-1))
    # plt.subplot2grid((2,5), (1,1)),plt.imshow(dilated, cmap = 'gray')
    # plt.title('Dilated'), plt.xticks([]), plt.yticks([])

    # img_with_contours = img_copy.copy()
    # cv2.drawContours(img_with_contours, all_contours, -1, (0,255,60), 3)
    # plt.subplot2grid((2,5), (1,2)),plt.imshow(img_with_contours)
    # plt.title('All Contours'), plt.xticks([]), plt.yticks([])

    # img_with_squares = img_copy.copy()
    # cv2.drawContours(img_with_squares, squares, -1, (0,255,60), 3)
    # plt.subplot2grid((2,5), (1,3)),plt.imshow(img_with_squares)
    # plt.title('All Rectangles'), plt.xticks([]), plt.yticks([])

    # img_with_top_square = img_copy.copy()
    # cv2.drawContours(img_with_top_square, [sorted_squares[0]], -1, (0,255,60), 3)
    # plt.subplot2grid((2,5), (1,4)),plt.imshow(img_with_top_square)
    # plt.title('Top Ranked Shape'), plt.xticks([]), plt.yticks([])

    # #plt.show()
    # img_with_top_square = img_copy.copy()
    # cv2.drawContours(img_with_top_square, [sorted_squares[0]], -1, (0,255,60), 3)
    #cv2.imshow('b', img_with_top_square)

    rect = cv2.minAreaRect(sorted_squares[0])
    #print('$$$$$$$$$$$$$$$$$\n',rect)
    img_croped = crop_minAreaRect(img_copy, rect)
     
    cv2.imwrite('croped.png', img_croped)
    #img_croped = fix_orientation(img_croped)
    print("CRP")
    #cv2.imwrite('croped', img_croped)
    #roi = rm_logo(img_croped)
    #cv2.imwrite('fixed.png', img_croped)
    return img_croped
    # cv2.imshow('c', img_croped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def extract_roi_3(img):
    #img = cv2.imread('received.png',)
    img = cv2.resize(img, (600,480))
    img_copy = img.copy()[:,:,::-1] # color channel plotting mess http://stackoverflow.com/a/15074748/2256243
    height = img.shape[0]
    width = img.shape[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # matrix of ones

    squares = []
    all_contours = []

    for gray in cv2.split(img):
        dilated = cv2.dilate(src = gray, kernel = kernel, anchor = (-1,-1))

        blured = cv2.medianBlur(dilated, 7)

        small = cv2.pyrDown(blured, dstsize = (int(width / 2), int(height / 2)))
        oversized = cv2.pyrUp(small, dstsize = (width, height))

        for thrs in range(0, 255, 26):
            if thrs == 0:
                edges = cv2.Canny(oversized, threshold1 = 0, threshold2 = 50, apertureSize = 3)
                next = cv2.dilate(src = edges, kernel = kernel, anchor = (-1,-1))
            else:
                retval, next = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)

            contours = cv2.findContours(next, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if imutils.is_cv2() else contours[1]
            # how are the contours sorted? outwards to inwards? would be interesting to do a PVE
            # sort of thing where the contours within a contour (and maybe see an elbow plot of some sort)
            for cnt in contours:
                all_contours.append(cnt)
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)

    sorted_squares = sorted(squares, key=lambda square: rank(square, img))

    
    rect = cv2.minAreaRect(sorted_squares[0])
    pts = crop_minAreaRect2(img_copy, rect)
    print("ptssss", pts)
    pts_1_1 = pts[1][1]
    pts_0_1 = pts[0][1]
    pts_1_0 = pts[1][0]
    pts_2_0 = pts[2][0]

    center , wh, theta = rect
    x,y = center
    w,h = wh   
    roi_metrices = {'x':x, 'y':y,'w':w,'h':h, 'theta':theta, 'pts_1_1':int(pts_1_1), 'pts_0_1':int(pts_0_1), 'pts_1_0': int(pts_1_0), 
    'pts_2_0':int(pts_2_0)}

    # img_cropped = img_rotated[pts[1][1]:pts[0][1],
    #                    pts[1][0]:pts[2][0]]
    return roi_metrices
    