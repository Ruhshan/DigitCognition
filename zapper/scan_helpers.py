from imageprocessing.datatoimage import make_image
from imageprocessing.roi_extraction import  extract_roi_2
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
        try:
            roi = extract_roi_2(img)
            
            result = detect(roi) 
            
            #write_characters(roi)

        except:
            result = "----------------"
        # # When roi is extracted its a 2d array 
        
    return result

