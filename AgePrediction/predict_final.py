import sys
from os import getcwd
from os import listdir
from os.path import isfile, join
import numpy as np
import nibabel as nib
from sklearn.svm.classes import SVC
import re
from scipy import linalg as LA
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    '''
    Helper function to sort files by their name numerically, so that file_7 comes before file_10
    '''
    #Spliting by numeric characters present
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def readVoxels(path):
    '''
    Read voxel values from nii images. The max voxel value present in any train image is 4419.
    Hence, counting the number of different voxels with intensities in between 0 and 4419 and using 
    that as features.
    Returns an array of features for all images present in the path passed as argument.
    '''
    l = np.empty((0,4419))
    for f in path:
        print(f)
        img = nib.load(f)
        arr = img.get_data()
        #For each dimension, picking the range where maximum voxel intensity is present. 
        #For most of the training images, it is within cude selected below
        arr = arr[40:100,100:160,50:110]
        flat = arr.reshape((60*60*60))
        fea = np.zeros((4419))
        
        #Counting the number of voxels for each intensity value present in the cube selected above
        for val in flat:
            if val <= 4418:
                fea[val] = fea[val] + 1

        l = np.append(l, fea.reshape((1, 4419)), axis=0)
    return l

if __name__ == '__main__':
    '''
    Parameters to pass: 
    First - Path to folder containing set_train, set_test and targets.csv
    Second - Path to folder where final csv file to be stored with name Dota2Prediction.csv
    '''
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]

    #train set
    trainDir = inputDir + "\\set_train"
    trainFiles = sorted([join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))], key=numericalSort)
 
    #Build feature vectors for training images
    samples = readVoxels(trainFiles)
    print(len(samples))

    targetsPath = inputDir + "\\targets.csv" 
    targets = np.recfromcsv(targetsPath, delimiter=',', names=['a', 'b', 'c'])

    #Read age labels of training images
    labels = []
    for t in targets:
        labels.append(t[0])

    #Training LASSO regressor, alpha value tuned to produce best result when used alone on the test set
    regrL = linear_model.Lasso(alpha=15.0)
    regrL.fit(samples, labels)

    #Training Ridge regressor, alpha value tuned to produce best result when used alone on the test set
    regrR = linear_model.Ridge(alpha=1e-13, normalize=True)
    regrR.fit(samples, labels)

    #Training SVM with linear kernel
    regrS = SVC(kernel='linear')
    regrS.fit(samples, labels)

    #Training Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(samples, labels)

    #test set
    testDir = inputDir + "\\set_test"
    testFiles = sorted([join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))], key=numericalSort)

    #Read feature vectors of test images
    testSamples = readVoxels(testFiles)
    print(len(testSamples))

    #2D array to report final prediction in format (ID,Prediction)
    final = [[0 for j in range(2)] for i in range(139)]
    final[0][0] = 'ID'
    final[0][1] = 'Prediction'
    id = 1

    #Predict age of test image using each of the 4 models trained above
    for item in testSamples:
        predictionL = regrL.predict(item)
        predictionR = regrR.predict(item)
        predictionS = regrS.predict(item)
        predictionRfc = rfc.predict(item)

        final[id][0] = id
        #Taking the average of each of the model predictions as final age prediction
        final[id][1] = (predictionL[0]+predictionR[0]+predictionS[0]+predictionRfc[0])//4
        id = id + 1

    #Save csv file in the output directory provided as argument with name Dota2Prediction.csv
    np.savetxt(outputDir + "\\Dota2Prediction.csv", 
           final,
           delimiter=',', 
           fmt='%s'
           )
    print("Finished!")