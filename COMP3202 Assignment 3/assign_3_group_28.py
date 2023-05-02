#
#  Assignment 3
#
#  Group 28:
#  <Brandon Cuza> <bhc107@mun.ca>

####################################################################################
# Imports
####################################################################################
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, make_scorer, precision_score, recall_score

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################

#   Function to load training data from appropriate .csv files in either the same directory as this
#   file, or in the supplied directory
def loadTrainData(dir_path=None):
    dir_path = dir_path or os.path.dirname(__file__)
    d0 = pd.read_csv(os.path.join(dir_path, 'train.fdg_pet.sNC.csv'), header=None, names=['ctx-lh-inferiorparietalm', 'ctx-lh-inferiortemporal', 'ctx-lh-isthmuscingulate', 'ctx-lh-middletemporal', 'ctx-lh-posteriorcingulate', 'ctx-lh-precuneus', 'ctx-rh-isthmuscingulate', 'ctx-rh-posteriorcingulate', 'ctx-rh-inferiorparietal', 'ctx-rh-middletemporal', 'ctx-rh-precuneus', 'ctx-rh-inferiortemporal', 'ctx-lh-entorhinal', 'ctx-lh-supramarginal'])
    d1 = pd.read_csv(os.path.join(dir_path, 'train.fdg_pet.sDAT.csv'), header=None, names=['ctx-lh-inferiorparietalm', 'ctx-lh-inferiortemporal', 'ctx-lh-isthmuscingulate', 'ctx-lh-middletemporal', 'ctx-lh-posteriorcingulate', 'ctx-lh-precuneus', 'ctx-rh-isthmuscingulate', 'ctx-rh-posteriorcingulate', 'ctx-rh-inferiorparietal', 'ctx-rh-middletemporal', 'ctx-rh-precuneus', 'ctx-rh-inferiortemporal', 'ctx-lh-entorhinal', 'ctx-lh-supramarginal'])
    d0['Outcome'] = 0
    d1['Outcome'] = 1
    trainingSet = pd.concat([d0, d1], ignore_index=True)
    return trainingSet

#   Function to load test data from appropraite .csv files in either the same directory as this
#   file, or in the supplied directory
def loadTestData(dir_path=None):
    dir_path = dir_path or os.path.dirname(__file__)
    d0 = pd.read_csv(os.path.join(dir_path, 'test.fdg_pet.sNC.csv'), header=None, names=['ctx-lh-inferiorparietalm', 'ctx-lh-inferiortemporal', 'ctx-lh-isthmuscingulate', 'ctx-lh-middletemporal', 'ctx-lh-posteriorcingulate', 'ctx-lh-precuneus', 'ctx-rh-isthmuscingulate', 'ctx-rh-posteriorcingulate', 'ctx-rh-inferiorparietal', 'ctx-rh-middletemporal', 'ctx-rh-precuneus', 'ctx-rh-inferiortemporal', 'ctx-lh-entorhinal', 'ctx-lh-supramarginal'])
    d1 = pd.read_csv(os.path.join(dir_path, 'test.fdg_pet.sDAT.csv'), header=None, names=['ctx-lh-inferiorparietalm', 'ctx-lh-inferiortemporal', 'ctx-lh-isthmuscingulate', 'ctx-lh-middletemporal', 'ctx-lh-posteriorcingulate', 'ctx-lh-precuneus', 'ctx-rh-isthmuscingulate', 'ctx-rh-posteriorcingulate', 'ctx-rh-inferiorparietal', 'ctx-rh-middletemporal', 'ctx-rh-precuneus', 'ctx-rh-inferiortemporal', 'ctx-lh-entorhinal', 'ctx-lh-supramarginal'])
    d0['Outcome'] = 0
    d1['Outcome'] = 1
    testSet = pd.concat([d0, d1], ignore_index=True)
    return testSet

#   Function to calculate the sensitivity of a model based on the passed
#   in actual labels y and the passed in predicted labels y_pred.
def sensitivity_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp/(tp + fn)
    return sensitivity

#   Function to calculate the specificity of a model based on the passed
#   in actual labels y and the passed in predicted labels y_pred.
def specificity_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn/(tn + fp)
    return specificity

#   Function to display a graph plotting the inputted x-values and y-values
#   and labels the graph's y-axis based on the handed in ylabel
def displayGraph(xvalues ,yvalues, ylabel):
    plt.plot(xvalues, yvalues, linestyle = '--', marker = 'o')
    plt.xscale("log", base = 2)
    plt.ylabel('%s' % ylabel)
    plt.xlabel('C')
    plt.title('Linear SVM %s as function of C' % ylabel)
    plt.show()

#   Function that takes a string specifying the expected kernel type ('linear' for linear,
#   'poly' for polynomial, 'rbf' for radial basis function) and trains a
#   SVM with that kernel type on the training dataset and evaluates it using the
#   test dataset.
#
#   Note to grader: when running this function for Q2 (i.e. on the polynomial SVM), my computer
#   took approx. 1h30min to process the grid search with n_jobs = 2. This time can be reduced by
#   setting n_jobs higher, however I am unsure if GridSearchCV is thread-safe as when I set the
#   value of n_jobs to -1 (i.e. use all available threads) it would fail to run Q3 afterwords.
def classify(kernel):
    print('Performing classification...')

    numJobs = 2;

    svm = SVC(kernel = kernel)
    trainY = loadTrainData()['Outcome']
    trainX = loadTrainData().drop('Outcome', axis='columns')
    testY = loadTestData()['Outcome']
    testX = loadTestData().drop('Outcome', axis='columns')
    
    if kernel == 'linear':
        
        cvals = [2**x for x in range(-5, 16)] # Chosen as per https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf pg.5 section 3.2
        param_grid = {'C' : cvals}
        grid = GridSearchCV(svm, param_grid, scoring = {'Accuracy': make_scorer(accuracy_score),
                                                        'Sensitivity': make_scorer(sensitivity_score),
                                                        'Specificity': make_scorer(specificity_score),
                                                        'Precision': make_scorer(precision_score),
                                                        'Recall': make_scorer(recall_score),
                                                        'Balanced Accuracy': make_scorer(balanced_accuracy_score)
                                                        }
                            , n_jobs = numJobs, refit = 'Balanced Accuracy', return_train_score = True)
        grid.fit(trainX, trainY)
        pred = grid.predict(testX)
        print("Best C found to be %i" % grid.best_params_.get('C'))
        print("Linear SVM - Accuracy Score: %.4f" % accuracy_score(testY, pred))
        print("Linear SVM - Sensitivity Score: %.4f" % recall_score(testY, pred))
        print("Linear SVM - Specificity Score: %.4f" % specificity_score(testY, pred))
        print("Linear SVM - Precision Score: %.4f" % precision_score(testY, pred))
        print("Linear SVM - Recall Score: %.4f" % recall_score(testY, pred))
        print("Linear SVM - Balanced Accuracy Score: %.4f\n" % balanced_accuracy_score(testY, pred))
        displayGraph(cvals, grid.cv_results_.get('mean_test_Accuracy'), 'Accuracy')
        displayGraph(cvals, grid.cv_results_.get('mean_test_Sensitivity'), 'Sensitivity')
        displayGraph(cvals, grid.cv_results_.get('mean_test_Specificity'), 'Specificity')
        displayGraph(cvals, grid.cv_results_.get('mean_test_Precision'), 'Precision')
        displayGraph(cvals, grid.cv_results_.get('mean_test_Recall'), 'Recall')
        displayGraph(cvals, grid.cv_results_.get('mean_test_Balanced Accuracy'), 'Balanced Accuracy')

    if kernel == 'poly':
        
        cvals = [2**x for x in range(-5, 16)] # Chosen as per https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf pg.5 section 3.2
        dvals = [0, 1, 2, 3, 4] # Range chosen as each higher degree adds exponentially more processing time
        param_grid = {'C' : cvals,
                      'degree' : dvals}
        grid = GridSearchCV(svm, param_grid, scoring = 'balanced_accuracy', n_jobs = numJobs)
        grid.fit(trainX, trainY)
        pred = grid.predict(testX)
        print("Best C found to be %i" % grid.best_params_.get('C'))
        print("Best d found to be %i" % grid.best_params_.get('degree'))
        print("Polynomial kernel SVM - Accuracy Score: %.4f" % accuracy_score(testY, pred))
        print("Polynomial kernel SVM - Sensitivity Score: %.4f" % sensitivity_score(testY, pred))
        print("Polynomial kernel SVM - Specificity Score: %.4f" % specificity_score(testY, pred))
        print("Polynomial kernel SVM - Precision Score: %.4f" % precision_score(testY, pred))
        print("Polynomial kernel SVM - Recall Score: %.4f" % recall_score(testY, pred))
        print("Polynomial kernel SVM - Balanced Accuracy Score: %.4f\n" % balanced_accuracy_score(testY, pred))

    if kernel == 'rbf':
        
        cvals = [2**x for x in range(-5, 16)] # Chosen as per https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf pg.5 section 3.2
        gvals = [2**x for x in range(-15, 4)] # Chosen as per https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf pg.5 section 3.2
        param_grid = {'C' : cvals,
                      'gamma' : gvals}
        grid = GridSearchCV(svm, param_grid, scoring = 'balanced_accuracy', n_jobs = numJobs)
        grid.fit(trainX, trainY)
        pred = grid.predict(testX)
        print("Best C found to be %i" % grid.best_params_.get('C'))
        print("Best γ found to be %.4f" % grid.best_params_.get('gamma'))
        print("RBF kernel SVM - Accuracy Score: %.4f" % accuracy_score(testY, pred))
        print("RBF kernel SVM - Sensitivity Score: %.4f" % sensitivity_score(testY, pred))
        print("RBF kernel SVM - Specificity Score: %.4f" % specificity_score(testY, pred))
        print("RBF kernel SVM - Precision Score: %.4f" % precision_score(testY, pred, pos_label='Outcome'))
        print("RBF kernel SVM - Recall Score: %.4f" % recall_score(testY, pred))
        print("RBF kernel SVM - Balanced Accuracy Score: %.4f" % balanced_accuracy_score(testY, pred))

#   Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
#   corresponding to each of the N_test features vectors in Xtest
#
#   Xtest       N_test x 14 matrix of test feature vectors
#
#   data_dir    full path to the folder containing the following files:
#               train.fdg_pet.sNC.csv, train.fdg_pet.sDAT.csv,
#               test.fdg_pet.sNC.csv, test.fdg_pet.sDAT.csv
def diagnoseDAT(Xtest, data_dir):
    svm = SVC(C = 1, kernel = 'poly', degree = 3)
    trainY = loadTrainData(data_dir)['Outcome']
    trainX = loadTrainData(data_dir).drop('Outcome', axis='columns')
    svm.fit(trainX, trainY)
    return svm.predict(Xtest)

    
def Q1_results():
    print('Generating results for Q1...')
    classify('linear')

def Q2_results():
    print('Generating results for Q2...')
    classify('poly')

def Q3_results():
    print('Generating results for Q3...')
    classify('rbf')

#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
