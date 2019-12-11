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
x = 0
inp = "grass/None/aircraft/grass/aircraft/grass/None/aircraft/grass/None/aircraft/grass/None/aircraft/bike/None/bike/None/bike/bike/None/bike/None/bike/None/car/None/car/car/None/car/None/None/None/car/car/None/None/car/car/None/None/None/None/cow/grass/None/cow/grass/cow/grass/cow/grass/None/None/face/None/None/None/face/None/None/face/face/None/None/face/face/None/None/grass/grass/grass/grass/None/grass/grass/grass/house/house/None/house/grass/grass/house/house/grass/house/None/house/grass/grass/tree/tree/None/tree/grass/None/tree/None/None/tree/None".split("/")
def image_segment(image):
    #image_slic = seg.slic(image,n_segments=1000)
    #seg_image = color.label2rgb(image_slic, image, kind='avg')
    #return seg_image
    return  segmentation_algorithms.makeThing(image)

def extract_feature_training(img, channel=3):
    feature_list = []
    global x
    global inp
    for seg in img:
        #label = input("Enter this training segment's classifier (enter 'None' if not semantically significant): ")
        label = inp[x]
        x += 1
        if label != "None":
            feature_list.append([seg, label])

    return np.array(feature_list)


def extract_feature(img, channel=3):
    feature_list = []
    for seg in img:
        feature_list.append([seg])

    return np.array(feature_list)


if __name__ == '__main__':
    directory = constants.TRAIN_DIR
    label_list = os.listdir(directory)
    svm = MultiLabelSVM()
    feature_label = []
    for i in range(len(label_list)):
        feature_label.append([])
        feature_label[i] = []
    scaler = sklearn.preprocessing.StandardScaler()

    X_list = []
    
    print("start to read training images...")
    for label in label_list:
        label_dir = directory + label
        image_list = [label_dir + '/' +  dir for dir in os.listdir(label_dir)]

        feature_list = []
        num = 0
        for image in image_list:
            num += 1
            if num > 5:
                break
            img = plt.imread(image)
            print("processing segmentation for %s..." % image)
            segment_image = image_segment(img)
            image_features = extract_feature_training(segment_image, channel = 1)
            for segment in image_features:
                feature_label[label_list.index(segment[1])].append(segment[0])
            
    for i in range(len(label_list)):
        X_list.append(np.vstack(feature_label[i]))

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
        #X_test = np.array(X_test).reshape(1,-1)
        #X_test = scaler.transform(X_test)
        print("start to predict %s" % image)
        y_pred = ""
        for segment in X_test:
            y_pred += " - " + svm.predict(np.array(X_test).reshape(1,-1))
        window.insert_image(img, label = y_pred)
        
    
    window.start()
    
