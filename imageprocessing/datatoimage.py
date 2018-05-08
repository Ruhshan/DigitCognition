import base64
import numpy as np
import cv2


def make_image(payload):
    """
    Tries to convert base64 encoded payload to opencv readable format

    Parameters
    ----------
    payload:    String
                base64 encoded image

    Returns:
    ----------
    0, img:     If conversion successful
    1, None:    If conversion fails
    """
    try:
        data = payload.split(',')[1]
        missing_padding = len(data) % 4

        if missing_padding != 0:
            data += b'=' * (4 - missing_padding)
        data = base64.b64decode(data)
        npimg = np.fromstring(data, dtype=np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        return 0, img
    except:
        return 1, None
