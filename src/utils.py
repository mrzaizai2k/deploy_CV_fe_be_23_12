import base64
from PIL import Image
import io
import numpy as np
import cv2

prefix = "data:image/png;base64"
prefix=''
# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string)[len(prefix):])
    img = Image.open(io.BytesIO(imgdata))
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img 


def imgToBase64(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    rawBytes = io.BytesIO()
    im = Image.fromarray(img.astype("uint8"))
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    return prefix+ base64.b64encode(rawBytes.read()).decode()