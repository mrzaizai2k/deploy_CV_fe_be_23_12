import sys
sys.path.append("")

import cv2
import numpy as np
from src.utils import  stringToRGB, imgToBase64
import json
import requests
import warnings
warnings.filterwarnings("ignore")
url = "http://127.0.0.1:5000/eyes"

def call_api(image):
    data = imgToBase64(image)
    res = requests.post(url, json ={"data":data})
    if res.status_code == 200:
        img_str = res.json()["data"]
        image = stringToRGB(img_str)
        return image
    else:
        return None

if __name__ == "__main__":
    cap =cv2.VideoCapture(0)
    cnt = 1000
    while  True and cnt > 0:
        cnt-=1
        ret, image = cap.read() 
        img = call_api(image)
        if img is not None:
            cv2.imshow("abc",img)
        
            k = cv2.waitKey(20) & 0xff
            if k == ord('q'):
                break
