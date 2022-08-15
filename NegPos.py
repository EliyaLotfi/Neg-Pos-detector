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

def load_data():
    data_list= []
    labels=[]

    for i,address in enumerate(glob.glob("smile_dataset\\*\\*")): 
        try:
            img = cv2.imread(address)
            img = detect_lips(img)
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
            
            data_list.append(img_r)

            label = address.split("\\")[1]
            labels.append(label)
            if i%100 == 0:
                print("statue: {}/3900 processed".format(i))
        except:
            print("Error")
            continue
    data_list=np.array(data_list)
    x_train,x_test,y_train,y_test=train_test_split(data_list,labels,test_size=0.2,random_state=0)
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test = load_data()

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:{:.2f}".format(accuracy*100))

dump(model,"smile1.z")