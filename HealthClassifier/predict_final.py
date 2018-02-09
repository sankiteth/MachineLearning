import sys
import os
from os import getcwd
from os import listdir
from os.path import isfile, join
import numpy as np
import nibabel as nib
from sklearn.svm.classes import SVC
import re
from scipy import linalg as LA
import scipy as sp
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    '''
    Helper function to sort files by their name numerically, so that file_7 comes before file_10
    '''
    #Spliting by numeric characters prsent
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def cubeVoxelsVar(path):
    '''
    Preprocesses the training files. Considers only arr[40:120,80:160,40:120]
    of voxels, slices them into cubes of dimension (10*10*10), and calculates 
    variance in each cube
    '''
    x_s = 40                                              #Best value 40 
    x_e = 120                                             #Best value 120
    y_s = 80                                              #Best value 80
    y_e = 160                                             #Best value 160
    z_s = 40                                              #Best value 40
    z_e = 120                                             #Best value 120
    cube_edge_x = 10                                      #Best value 10
    cube_edge_y = 10
    cube_edge_z = 10
    num_cubes = ((x_e - x_s) * (y_e - y_s) * (z_e - z_s)) // (cube_edge_x * cube_edge_y * cube_edge_z)
    feaMat = np.empty((0, num_cubes), float)

    # first path leads to a .DS_store file
    for f in path:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        # For each dimension, picking the range where maximum voxel intensity is present.
        # For most of the training images, it is within cube selected below
        arr = arr[x_s:x_e, y_s:y_e, z_s:z_e]
        arr = arr.reshape((x_e - x_s, y_e - y_s, z_e - z_s))
        
        fea = np.empty((num_cubes,), float)
        index = 0
        for i in range(0, x_e - x_s, cube_edge_x):
            for j in range(0, y_e - y_s, cube_edge_y):
                for k in range(0, z_e - z_s, cube_edge_z):
                    # Calculate variance in each smaller cube
                    fea[index] = np.var(arr[i:i+cube_edge_x-1, j:j+cube_edge_y-1, k:k+cube_edge_z-1])

                    index = index + 1
        feaMat = np.append(feaMat, fea.reshape((1, fea.shape[0])), axis=0)
    return feaMat

def feature_selection(X, y):
    '''
    Select 195 features with highest mutual information scores
    '''
    k_best = SelectKBest(mutual_info_classif, k=195).fit(X, y)
    return k_best

def logloss(act, pred):
    '''
    Calculate the log loss incurred for each prediction
    '''
    epsilon = 1e-15
    pred = max(epsilon, pred)
    pred = min(1-epsilon, pred)
    ll = act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred))
    ll = ll * -1.0
    return ll

def cv_split(totalSamples, totalLabels, K):
    '''
    takes total samples with labels
    returns (samples to train with labels, samples to validate with labels)
    K-fold cross validation with K = 10
    '''
    num = len(totalLabels)
    shuffle_index = np.random.permutation(num)
    shuffled_samples = totalSamples[shuffle_index]
    shuffled_labels = totalLabels[shuffle_index]

    fold_samples = []
    fold_labels = []

    for i in range(0, num, int(np.ceil(num/K)) ):
        fold_samples.append(shuffled_samples[i : i + int(np.ceil(num/K)) ])
        fold_labels.append(shuffled_labels[i : i + int(np.ceil(num/K)) ])

    return (fold_samples, fold_labels)

def cross_validate(samples, labels, outputDir):
    '''
    Function to perform K-fold cross validation
    '''
    # K(=10) FOLD CROSS VALIDATION
    K = 10
    fold_samples, fold_labels = cv_split(samples, np.array(labels), K)
    log_loss = [['Log Loss'],[]]
    total_ll = 0.0
    for fold in range(K):
        samples_chunk = fold_samples[:fold] + fold_samples[fold+1:]
        labels_chunk = fold_labels[:fold] + fold_labels[fold+1:]
    
        #Training L1 logistic regression
        logRegrL1 = linear_model.LogisticRegression(C=1, penalty='l1')
        logRegrL1.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )
    
        #Training SVM with linear kernel
        svmLin = SVC(kernel='linear', probability=True)
        svmLin.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )
    
        #Training Random Forest Classifier
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit( np.concatenate(samples_chunk, axis=0), np.concatenate(labels_chunk, axis = 0) )
    
        #TEST ON CROSS VALIDATION HOLD OUT SET
        val = [i for i in range(len(fold_labels[fold]))]
        id = 0
        for item in fold_samples[fold]:
            predictionL1 = logRegrL1.predict_proba(item)#first component is probability of 0 class, second is of class 1
            predictionSvmLin = svmLin.predict_proba(item)
            predictionRfc = rfc.predict_proba(item)
    
            #Taking the average of each of the model predictions as final health status prediction
            val[id] = (predictionL1[0][1] + predictionSvmLin[0][1] + predictionRfc[0][1])/3.0
            id = id + 1
    
        
        for i in range(len(fold_labels[fold])):
            total_ll += logloss(fold_labels[fold][i], val[i])
    
    
    log_loss[1] = total_ll/len(samples)
    #Save csv file in the output directory with name Dota2Val.csv
    np.savetxt(outputDir + "\\Dota2Val.csv", 
           log_loss,
           delimiter=',', 
           fmt='%s'
           )

def train_and_predict(samples, labels, feature_selector, inputDir, outputDir):
    #Training L1 logistic regression
        logRegrL1 = linear_model.LogisticRegression(C=1, penalty='l1')
        logRegrL1.fit(samples, labels)

        #Training SVM with linear kernel
        svmLin = SVC(kernel='linear', probability=True)
        svmLin.fit(samples, labels)

        #Training Random Forest Classifier
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(samples, labels)

        #test set
        testDir = inputDir + "\\set_test"
        testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)

        #Read feature vectors of test images
        testSamples = cubeVoxelsVar(testFiles)
        testSamples = feature_selector.transform(testSamples)
        print(len(testSamples))

        #2D array to report final prediction in format (ID,Prediction)
        final = [[0 for j in range(2)] for i in range(139)]
        final[0][0] = 'ID'
        final[0][1] = 'Prediction'
        id = 1

        #Predict health status of test image using each of the 3 models trained above
        for item in testSamples:
            predictionL1 = logRegrL1.predict_proba(item)#first component is probability of 0 class, second is of class 1
            predictionSvmLin = svmLin.predict_proba(item) 
            predictionRfc = rfc.predict_proba(item)

            final[id][0] = id
            #Taking the average of each of the model predictions as final health status prediction
            final[id][1] = (predictionL1[0][1] + predictionSvmLin[0][1] + predictionRfc[0][1])/3.0
            id = id + 1
    
        #Save csv file in the output directory with name final_sub.csv
        np.savetxt(outputDir + "\\final_sub.csv", 
               final,
               delimiter=',', 
               fmt='%s'
               )

if __name__ == '__main__':
    currentDir = os.path.dirname(os.path.realpath(__file__))

    inputDir = currentDir + "\\Data"
    outputDir = currentDir

    mode = 'test'

    #train set
    trainDir = inputDir + "\\set_train"
    totalTrainFiles = sorted([join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))], key=numericalSort)
 
    #Preprocesses images
    samples = cubeVoxelsVar(totalTrainFiles)

    print(len(samples))

    targetsPath = inputDir + "\\targets.csv" 
    targets = np.recfromcsv(targetsPath, delimiter=',', names=['a', 'b', 'c'])

    #Read brain health status of each training image
    labels = []
    for t in targets:
        labels.append(t[0])

    # Do feature selection
    feature_selector = feature_selection(samples, np.array(labels))
    samples = feature_selector.transform(samples)

    print("Number of features = {}".format(samples[0].shape))

    if mode == 'validate':
        cross_validate(samples, labels, outputDir)
        print("Validation complete!")

    else:
        train_and_predict(samples, labels, feature_selector, inputDir, outputDir)
        print("Finished!")