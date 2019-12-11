import os
import pickle
import constants
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    classification_report,
    multilabel_confusion_matrix,
)
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from multilabel_svm import MultiLabelSVM
from main import extract_feature, image_segment
import numpy as np

# from sklearn.svm._classes import SVC

# measure performance


def measure_performance(y_pred, y_test):
    print("Accuracy: {0:.3f}".format(accuracy_score(y_test, y_pred)), "\n")
    print(
        "F1_score: {0:.3f}".format(
            fbeta_score(y_test, y_pred, beta=1, average="micro")
        ),
        "\n",
    )
    print("Classification report")
    print(classification_report(y_test, y_pred), "\n")
    print(multilabel_confusion_matrix(y_test, y_pred), "\n")


def predict(X, svm):
    X_test = extract_feature(image_segment(X), channel=1)
    #print("start to predict %s" % X)
    y_pred = ""
    for segment in X_test:
        y_pred = svm.predict(np.array(segment).reshape(1, 3 * 3))
    return y_pred


def main():
    directory = constants.TEST_DIR
    labels_dir = constants.TRAIN_DIR
    label_list = os.listdir(labels_dir)
    svm = pickle.load(open("trained_model.p", "rb"))
    X = []
    y = []
    for path in os.listdir(directory):
        X.append(plt.imread(directory + path))
        classes = path.split("_")[:-1]
        y.append([int(l in classes) for l in label_list])
    y_preds = [predict(x, svm) for x in X]
    print(y_preds, y)
    measure_performance(y_preds, y)


#
#
# if __name__ == "__main__":
#    directory = constants.TRAIN_DIR
#    label_list = os.listdir(directory)
#    svm = MultiLabelSVM()
#    X_list = []
#    scaler = sklearn.preprocessing.StandardScaler()
#
#    print("start to read training images...")
#    for label in label_list:
#        label_dir = directory + label
#        image_list = [label_dir + "/" + dir for dir in os.listdir(label_dir)]
#
#        feature_list = []
#        for image in image_list:
#            img = plt.imread(image)
# results            print("processing segmentation for %s..." % image)
#            segment_image = image_segment(img)
#            image_features = extract_feature(segment_image, channel=1)
#            feature_list.append(image_features)
#
#        X_list.append(np.vstack(feature_list))
#
#    print("start to train SVM...")
#    for i in range(len(label_list)):
#        print("start to train SVM for '%s' class" % label_list[i])
#        X_train = X_list[i]
#        y_train = np.ones(X_train.shape[0])
#        for j in range((len(label_list))):
#            if j != i:
#                X_train = np.vstack([X_train, X_list[j]])
#                y_train = np.append(y_train, np.zeros(X_list[j].shape[0]))
#
#        # scaler = scaler.fit(X_train)
#        # X_train = scaler.transform(X_train)
#        svm.train(X_train, y_train, label_list[i])
#
#    try:
#        pickle.dump(svm, open("trained_model.p", "wb"))
#    except:
#        print("saving error.")
#
#    directory = constants.TEST_DIR
#    image_list = [directory + img for img in os.listdir(directory)]
#
#    window = ScrolledFrame(width=850, height=500, title="result")
#    window.pack(expand=True, fill="both")
#    width = window.IMAGE_WIDTH
#    height = window.IMAGE_HEIGHT
#    img_list = []
#
#    print("start to read testing images...")
#    for image in image_list:
#
#        img = ImageTk.PhotoImage(
#            Image.open(image).resize((width, height), Image.ANTIALIAS)
#        )
#        img_list.append(img)
#        print("processing segmentation for %s" % image)
#        X_test = extract_feature(image_segment(plt.imread(image)), channel=1)
#        X_test = np.array(X_test).reshape(1, -1)
#        # X_test = scaler.transform(X_test)
#        print("start to predict %s" % image)
#        y_pred = svm.predict(np.array(X_test).reshape(1, -1))
#        window.insert_image(img, label=y_pred)
#
#    window.start()


main()
