'''
Created on Dec 8, 2019

@author: Admin
'''
import numpy as np
from sklearn.svm import SVC


class SVM:
    def __init__(self, label):
        self.label = label
        self.clf = SVC(random_state = 0, gamma='auto')
    
    def fit(self, X, y):
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

class MultiLabelSVM:
    def __init__(self):
        self.svm_list = []
        
    def label_exists(self, label):
        if len(self.svm_list) == 0:
            return False
        
        for svm in self.svm_list:
            if svm.label == label:
                return True
            
        return False
    
    def train(self, X, y, label): 
        if self.label_exists(label):
            print("SVM classifier with label \'%s\' has been created." % label)
            return
        
        new_svm = SVM(label)
        new_svm.fit(X, y)
        self.svm_list.append(new_svm)

        
    def predict(self, X):
        label_list = []
        for svm in self.svm_list:
            y_pred = svm.predict(X)
            if np.any(y_pred):
                label_list.append(svm.label)
    
        return label_list
    
    def print_labels(self):
        print([svm.label for svm in self.svm_list])
            
if __name__ == '__main__':
    multi_svm = MultiLabelSVM()
    multi_svm .print_labels()
    
    
    
