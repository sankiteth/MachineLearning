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
from sklearn.feature_selection import f_classif
from sklearn.neighbors.kde import KernelDensity

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    '''
    Helper function to sort files by their name numerically, so that file_7 comes before file_10
    '''
    #Spliting by numeric characters prsent
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def cubeVoxelsVar_gender(path):
    '''
    Preprocesses the training files. Considers only arr[40:120,80:160,40:120]
    of voxels, slices them into cubes of dimension (8*8*18), and calculates mean and 
    variance in each cube
    '''
    x_s = 40                                              #Best value 40 
    x_e = 120                                             #Best value 120
    y_s = 80                                              #Best value 80
    y_e = 160                                             #Best value 160
    z_s = 40                                              #Best value 40
    z_e = 120                                             #Best value 120
    cube_edge_x = 8                                       #Best value 8
    cube_edge_y = 8
    cube_edge_z = 8
    num_cubes = ((x_e - x_s) * (y_e - y_s) * (z_e - z_s)) // (cube_edge_x * cube_edge_y * cube_edge_z)

    feaMat = np.empty((0, num_cubes), float)

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
                    mean = np.mean(arr[i:i+cube_edge_x-1, j:j+cube_edge_y-1, k:k+cube_edge_z-1])
                    vari = np.var(arr[i:i+cube_edge_x-1, j:j+cube_edge_y-1, k:k+cube_edge_z-1])
                    fea[index] = mean + vari
                    index += 1             

        feaMat = np.append(feaMat, fea.reshape((1, fea.shape[0])), axis=0)
    print("Feature extraction for gender done!")
    return feaMat

def cubeVoxelsVar_age(path):
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

    for f in path:
        #print(f)
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
                    variance = np.var(arr[i:i+cube_edge_x-1, j:j+cube_edge_y-1, k:k+cube_edge_z-1])
                    fea[index] = variance

                    index = index + 1
        feaMat = np.append(feaMat, fea.reshape((1, fea.shape[0])), axis=0)
    print("Feature extraction done!")
    return feaMat

def cubeVoxelsVar_health(path):
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

    for f in path:
        #print(f)
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
    print("Feature extraction done!")
    return feaMat

def feature_selection(X, y, K=195):
    '''
    Select K features with highest mutual information scores
    '''
    k_best = SelectKBest(mutual_info_classif, k=K).fit(X, y)
    return k_best

def train_and_predict(samples, labels, feature_selector, inputDir, outputDir):
    #test set
    testDir = inputDir + "\\set_test"
    testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)

    # Different features for gender
    testSamples_gender = cubeVoxelsVar_gender(testFiles)

    # Same features for age and health
    testSamples_age    = cubeVoxelsVar_age(testFiles)
    testSamples_health = testSamples_age 

    testSamples = [testSamples_gender, testSamples_age, testSamples_health]

    #2D array to report final prediction in format (ID,Prediction)
    final = [[0 for j in range(4)] for i in range(1 + 138*3)]
    final[0][0] = 'ID'
    final[0][1] = 'Sample'
    final[0][2] = 'Label'
    final[0][3] = 'Predicted'

    total_labels = ['gender', 'age', 'health']

    for label in range(3):
        print("Prediction label 1 started!")
        id_count = label
        #Training logistic regression
        logRegrL1 = linear_model.LogisticRegression()
        logRegrL1.fit(samples[label], labels[label])

        #Training SVM with linear kernel
        svmLin = SVC(kernel='linear')
        svmLin.fit(samples[label], labels[label])

        #Training Random Forest Classifier
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(samples[label], labels[label])

        print("Training complete!")
        
        # Do feature selection only for age and health
        if label == 0:
            testSamples_curr = testSamples[label]
        else:
            testSamples_curr = feature_selector[label].transform(testSamples[label])
        print(len(testSamples_curr))
        
        id = label+1

        #Predict gender, age and health status of test image using each of the 3 models trained above
        for sampleNum, sample in enumerate(testSamples_curr):
            predictionL1 = logRegrL1.predict(sample)
            predictionSvmLin = svmLin.predict(sample) 
            predictionRfc = rfc.predict(sample)

            final[id][0] = id_count
            final[id][1] = sampleNum
            final[id][2] = total_labels[label]

            votes = predictionL1[0] + predictionSvmLin[0] + predictionRfc[0]

            final[id][3] = 'TRUE' if votes >= 2.0 else 'FALSE'
            id = id + 3
            id_count = id_count + 3
        print('Prediction done!')
    
    #Save csv file in the output directory with name final_sub.csv
    np.savetxt(outputDir + "\\final_sub.csv", 
           final,
           delimiter=',', 
           fmt='%s'
           )

if __name__ == '__main__':
    print("Start!")
    currentDir = os.path.dirname(os.path.realpath(__file__))

    inputDir = currentDir + "\\Data"
    outputDir = currentDir

    #train set
    trainDir = inputDir + "\\set_train"
    totalTrainFiles = sorted([join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))], key=numericalSort)
 
    #Preprocesses images
    samples_gender = cubeVoxelsVar_gender(totalTrainFiles)
    samples_age    = cubeVoxelsVar_age(totalTrainFiles)
    samples_health = samples_age

    targetsPath = inputDir + "\\targets.csv" 
    targets_binary = np.recfromcsv(targetsPath, delimiter=',', names=['a', 'b', 'c'])

    #Read labels
    labels_gender = []
    labels_age = []
    labels_health = []
    for t in targets_binary:
        labels_gender.append(t[0])
        labels_age.append(t[1])
        labels_health.append(t[2])

    labels = (labels_gender, labels_age, labels_health)

    # Do feature selection
    feature_selector_gender = feature_selection(samples_gender, np.array(labels_gender))
    feature_selector_age    = feature_selection(samples_age,    np.array(labels_age))
    feature_selector_health = feature_selection(samples_health, np.array(labels_health))

    feature_selector = (feature_selector_gender, feature_selector_age, feature_selector_health)

    samples_age    = feature_selector_age.transform(samples_age)
    samples_health = feature_selector_health.transform(samples_health)

    samples = (samples_gender, samples_age, samples_health)

    train_and_predict(samples, labels, feature_selector, inputDir, outputDir)
    print("Finished!")