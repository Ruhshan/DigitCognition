import cv2
import numpy as np
import imutils
from imageprocessing.contours_functions import angle_cos, rank, mask_image

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
def extract_roi_2():
    img = cv2.imread('received.png',)
    #img = cv2.resize(img, (600,480))
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

    rect = cv2.minAreaRect(sorted_squares[0])
    #print('$$$$$$$$$$$$$$$$$\n',rect)
    img_croped = crop_minAreaRect(img_copy, rect)
     
    cv2.imwrite('croped.png', img_croped)
    #img_croped = fix_orientation(img_croped)
    print("CRP")
    #cv2.imwrite('croped', img_croped)
    #roi = rm_logo(img_croped)
    #cv2.imwrite('fixed.png', img_croped)
    
extract_roi_2()