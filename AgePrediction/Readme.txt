Team Name: DOTA2

Authors: Ankit Srivastava, Parijit Kedia, Simon Hofer

How to run:
To run the code, we need to pass 2 command line arguments:
    First - Path to folder containing set_train, set_test and targets.csv
    Second - Path to folder where final csv file to be stored with name Dota2Prediction.csv

Summary of Approach:
1. Preprocessing:
We looked at the voxel intensities to be used as features. Along all the axes, the voxel values ranged between 0 and 4418
for all the training as well as test images. Also, we considered the voxel intensities along each axes and found that
"arr[40:100,100:160,50:110]" has maximum voxel intensity. We ran a few times with different ranges and found that these (60*60*60) pixels
were producing the best result, both on the given test set as well as a validation set that we created from training test.

2. Feature selection:
As mentioned above, for each image, we created a feature vector of 4419 elements, where each element contains the number of voxels
with that voxel intensity.

3. Training and Final prediction:
We trained 4 different models, and then averaged their predictions to predict the final age. This reduced the variance of the result, without increasing
the bias. The averaged model performed better than all the moodels did individually, as expected. The models that we used are:
	a. LASSO Linear Regressor with alpha=15.0
	b. Ridge Linear Regressor with alpha=1e-13
	c. SVM with linear kernel
	d. Random Forest Classifier with 100 estimators.

To choose the parameters (regularization parameter, kernel, number of estimators), we ran the models separately on the training set
to produce their individual best result in terms of MSE and ensured that the whole program runs in reasonable time (within 10 mins on 4gb RAM machine).
Because of the Random Forrest Classifier, the final result has an element of non-determinism, but the variation is very less on multiple runs.

4. Validation:
We held out nearly 20% training images (30 in number) for cross validation (only 1 fold). So, the models were trained on 248 images, tested on 30 images,
and then the final model that we came up with was trained on 100% of the data.

 