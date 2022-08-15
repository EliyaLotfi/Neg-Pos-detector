import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "3"

import cv2
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump,load
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from joblib import dump
from mtcnn import MTCNN

detector = MTCNN()

def detect_lips(img):
    rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)         
    out = detector.detect_faces(rgb_img)[0]
    x,y,w,h=out["box"]
    nose=out["keypoints"]['nose'][1]
    mouth_left=out["keypoints"]['mouth_left'][0]
    mouth_right=out["keypoints"]['mouth_right'][0]
    img = img[nose:y+h,mouth_left:mouth_right]
    return img