import pickle
from multilabel_svm import MultiLabelSVM
import constants
from gui import ScrolledFrame
import sklearn.preprocessing
from PIL import ImageTk, Image
import numpy as np
import segmentation_algorithms
import scipy
import os
import matplotlib.pyplot as plt


def image_segment(image):
    #image_slic = seg.slic(image,n_segments=1000)
    #seg_image = color.label2rgb(image_slic, image, kind='avg')
    #return seg_image
    return  np.array(segmentation_algorithms.makeThing(image))

def extract_feature(img, channel=3):
    copy = img.reshape(img.shape[0]*img.shape[1],channel)
    feature_list = []
    
    for i in range(channel):
        feature_list.append(np.mean(copy[:,i]))
        feature_list.append(np.var(copy[:,i]))
        feature_list.append(scipy.stats.skew(copy[:,i]))
        feature_list.append(scipy.stats.moment(copy[:,i]))
        
    return np.array(feature_list)

if __name__ == '__main__':
    svm = pickle.load(open('trained_model.p', "rb" ))
    
    directory = constants.TEST_DIR + '../tmp/'
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