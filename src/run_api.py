import sys
sys.path.append("")
import mediapipe as mp
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
from src.utils import stringToRGB, imgToBase64
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import requests
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_img = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

### 5. Building a CNN model
class EyeDetectionModelV0(nn.Module):
    # Building a constructor
    def __init__(self):
        super().__init__()

        # CNN block 1
        self.conv_layer_1 = nn.Sequential(
                nn.Conv2d(in_channels=3,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        )

        # CNN block 2
        self.conv_layer_2 = nn.Sequential(
                nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        )

        # CNN block 3
        self.conv_layer_3 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        )

        # Fully connected layer
        self.fc_layer_1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=128*4*4,
                                    out_features=512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(p=0.5)
        )

        self.fc_layer_2 = nn.Sequential(
                nn.Linear(in_features=512,
                                    out_features=256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5)
        )

        self.fc_layer_3 = nn.Sequential(
                nn.Linear(in_features=256,
                                    out_features=64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.5)
        )

        # Classifier layer
        self.classifier = nn.Linear(in_features=64,
                                                                out_features=1)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        # print(x.shape)
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        x = self.fc_layer_3(x)
        x = self.classifier(x)
        return x
    
def plot_landmark(img_base, facial_area_obj, results):
    all_lm = []
    img = img_base.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = results.multi_face_landmarks[0]
    for source_idx, target_idx in facial_area_obj:
        source = landmarks.landmark[source_idx] # take the coordinations of results base on const
        target = landmarks.landmark[target_idx]

        relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
        relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
        all_lm.append(relative_source)
        all_lm.append(relative_target)

    all_lm = sorted(all_lm, key = lambda a: (a[0]))
    x_min, x_max = all_lm[0][0], all_lm[-1][0]
    all_lm = sorted(all_lm, key = lambda a: (a[1]))
    y_min, y_max = all_lm[0][1], all_lm[-1][1]

    img_ = img[y_min-10:y_max+1+10, x_min-10:x_max+1+10]
    return img_, [(x_min-10, y_min-10), (x_max+10, y_max+10)]


def predict(img, model):
    img = transform_img(img)
    img = torch.unsqueeze(img, 0).type(torch.float32).to(device)

    model.eval()
    with torch.inference_mode():
        output = model(img)

    img_preds = torch.round(torch.sigmoid(output)).squeeze(dim=0).type(torch.int32).item()
    p = label2id[img_preds]
    return p

def process(image):
    global count
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # cvt BRG to RGB

    if(results.multi_face_landmarks):
        # Left eye
        l_eyebrow, coor1 = plot_landmark(image, mp_face_mesh.FACEMESH_LEFT_EYE, results)
        img = Image.fromarray(l_eyebrow)
        pred = predict(img, model)

        if(str(pred) == "Close"):
            check_l = 1
        else:
            check_l = 0
        status_l = str(pred)
        pred = status_l
        # print(f"Left eye: {pred}")
        cv2.putText(image, str(pred), (coor1[0][0], coor1[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        cv2.rectangle(image, coor1[0], coor1[1], (255,0,0), 1)

        # Right eye
        r_eyebrow, coor2 = plot_landmark(image, mp_face_mesh.FACEMESH_RIGHT_EYE,results)
        img = Image.fromarray(r_eyebrow)
        pred = predict(img, model)

        if(str(pred) == "Close"):
            check_r = 1
        else:
            check_r = 0
        status_r = str(pred)
        pred = status_r
        # print(f"Right eye: {pred}")
        cv2.putText(image, str(pred), (coor2[0][0], coor2[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
        cv2.rectangle(image, coor2[0], coor2[1], (255,0,0), 1)

        if(check_l == 1 and check_r == 1):
            count += 1
        else:
            count = 0

        if(count > 40):
            cv2.rectangle(image, (0,0), (image.shape[1], image.shape[0]), (0,0,255), 5)
            cv2.putText(image, "Attention warning!!",
                                (int(image.shape[1]/3), image.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

    return image

@app.route("/eyes",methods=["POST"])
def run_api():
    base64_img = request.json["data"]

    # print(base64_img)
    image = stringToRGB(base64_img)
    image = process(image)
    base64_img = imgToBase64(image)
    return jsonify({"data" : base64_img})

if __name__ == "__main__":
    model = EyeDetectionModelV0()
    model.load_state_dict(torch.load("model/eye_model_28_10.pth", map_location=torch.device(device)))
    model.to(device)

    mp_face_mesh = mp.solutions.face_mesh


    # Encoder label
    label2id = {
            0: 'Close',
            1: 'Open',
    }
    status_l = ""
    status_r = ""
    count = 0
    check_l = 0
    check_r = 0
    app.run(debug=True)