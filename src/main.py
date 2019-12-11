from multilabel_svm import MultiLabelSVM
from gui import ScrolledFrame
import os 
import constants
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
import sklearn.preprocessing
import skimage.segmentation as seg
import skimage.color as color
import segmentation_algorithms
import pickle

def image_segment(image):
    #image_slic = seg.slic(image,n_segments=1000)
    #seg_image = color.label2rgb(image_slic, image, kind='avg')
    #return seg_image
    return  segmentation_algorithms.makeThing(image)

def extract_feature(img, channel=3):
    feature_list = []
    for seg in img:
        feature_list.append([seg])

        
    return np.array(feature_list)
    

if __name__ == '__main__':
    directory = constants.TRAIN_DIR
    label_list = os.listdir(directory)
    svm = MultiLabelSVM()
    X_list = []
    scaler = sklearn.preprocessing.StandardScaler()
    
    print("start to read training images...")
    for label in label_list:
        label_dir = directory + label
        image_list = [label_dir + '/' +  dir for dir in os.listdir(label_dir)]

        feature_list = []
        for image in image_list:
            img = plt.imread(image)
            print("processing segmentation for %s..." % image)
            segment_image = image_segment(img)
            image_features = extract_feature(segment_image, channel = 1)
            feature_list.append(image_features)
            
        X_list.append(np.vstack(feature_list))

    print("start to train SVM...")
    for i in range(len(label_list)):
        print("start to train SVM for \'%s\' class" % label_list[i])
        X_train = X_list[i]
        y_train = np.ones(X_train.shape[0])
        for j in range((len(label_list))):
            if j != i:
                X_train = np.vstack([X_train, X_list[j]])
                y_train = np.append(y_train, np.zeros(X_list[j].shape[0]))
        
        #scaler = scaler.fit(X_train)
        #X_train = scaler.transform(X_train)
        svm.train(X_train, y_train, label_list[i])
    
    try:
        pickle.dump(svm, open( "trained_model.p", "wb" ))
    except:
        print("saving error.")
    
    
    directory = constants.TEST_DIR
    image_list = [directory + img for img in os.listdir(directory)]
        
        
    window = ScrolledFrame(width=850, height=500, title='result')
    window.pack(expand=True, fill='both')
    width = window.IMAGE_WIDTH
    height = window.IMAGE_HEIGHT
    img_list = []    
    
    print("start to read testing images...")    
    for image in image_list:
        
        img = ImageTk.PhotoImage(Image.open(image).resize((width, height), Image.ANTIALIAS)) 
        img_list.append(img)
        print("processing segmentation for %s" % image)
        X_test = extract_feature(image_segment(plt.imread(image)), channel = 1)
        X_test = np.array(X_test).reshape(1,-1)
        #X_test = scaler.transform(X_test)
        print("start to predict %s" % image)
        y_pred = svm.predict(np.array(X_test).reshape(1,-1))
        window.insert_image(img, label = y_pred)
        
    
    window.start()
    