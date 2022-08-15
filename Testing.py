import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "3"

import cv2
import glob
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
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

clf = load("smile1.z")

for i,address in enumerate(glob.glob("tests\\*")): 
    img1 = cv2.imread(address)
    img = detect_lips(img1)
    if img is None:
        continue
    m,n,c= img.shape
    img_r=np.reshape(img,(m,n*c))
    pca=PCA(n_components=5).fit(img_r)
    img_r=pca.transform(img_r)       
    img_r=img_r.flatten()
    x=img_r.size
    c=x//5
    img_r=np.reshape(img_r,(5,c))
    pca=PCA(n_components=5).fit(img_r)
    img_r=pca.transform(img_r)
    x= img_r.max()
    img_r=img_r/x
    img_r=img_r.flatten()
    pred=clf.predict(np.array(img_r).reshape(1,-1))[0]
    cv2.putText(img1,pred,(40,40),cv2.FONT_HERSHEY_SIMPLEX,1.9,(255,0,0),10)
    cv2.imshow("image",img1)
    cv2.waitKey(0)
cv2.destroyAllWindows()