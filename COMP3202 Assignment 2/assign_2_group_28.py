#
#  Assignment 2
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import r2_score, make_scorer

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
params = 8
trainDataSize = 800
testDataSize = 100
folds = 5

#   Function to load training data from a .csv file in either the same directory as this
#   file, or in the supplied directory
def loadTrainData(dir_path=None):
    dir_path = dir_path or os.path.dirname(__file__)
    trainSet = pd.read_csv(os.path.join(dir_path, 'train.csv'), header=0)
    return trainSet

#   Function to load test data from a .csv file in either the same directory as this
#   file, or in the supplied directory
def loadTestData(dir_path=None):
    dir_path = dir_path or os.path.dirname(__file__)
    testSet = pd.read_csv(os.path.join(dir_path, 'test.csv'), header=0)
    return testSet
#   Functino to determine the residual standard error (RSE) of the handed in real
#   set of outcomes and set of predictions
def RSE(real, pred):
    real = real.to_numpy()
    df = pred.size - (params + 1)
    sum = 0
    for i in range(pred.size):
        diff = real[i]-pred[i]
        diffsq = diff**2
        sum +=diffsq
    temp = sum/df
    RSE = math.sqrt(temp)
    return RSE

#   Function to display a graph plotting the inputted x-values and y-values
#   and titles the graph based on the handed in model
def displayGraph(xvalues ,yvalues, model):
    plt.plot(xvalues, yvalues)
    plt.ylabel('R^2')
    plt.xlabel('alpha')
    plt.title(f'{model} performance as function of alpha')
    plt.ticklabel_format(useOffset=False)
    plt.show()

#   Workhorse function that does all the resression for the assignment
#   handed in values determine its procedure
def regression(trainDataX, trainDataY, testDataX=None, testDataY=None, model="linear", trainType="validation"):
    print('Performing regression...')

    # For Q1
    if (model == "linear"):
        regr = LinearRegression()
    # For Q2
    elif (model == "ridge"):
        regr = Ridge()
        xvals = list(np.arange(25, 10001, 25))
        param_grid = {'alpha' : xvals}
        grid = GridSearchCV(regr, param_grid, cv=folds, scoring="r2", return_train_score=True)
        grid.fit(trainDataX, trainDataY)
        best_alpha = grid.best_params_.get("alpha")
        regr = Ridge(alpha=best_alpha)
        regr.fit(trainDataX, trainDataY)
        pred = regr.predict(testDataX)
        print("Best alpha found to be %i" % best_alpha)
        print("Ridge, Grid Search Cross-validation - Residual square error: %.5f" % RSE(testDataY, pred))
        print("Ridge, Grid Search Cross-validation - Coefficient of determination: %.5f" % r2_score(testDataY, pred))
        displayGraph(xvals, grid.cv_results_.get("mean_test_score"), 'Ridge')
    # For Q3
    elif (model == "lasso"):
        regr = Lasso()
        xvals = list(np.arange(0.00025, 0.10001, 0.00025))
        param_grid = {'alpha' : xvals}
        grid = GridSearchCV(regr, param_grid, cv=folds, scoring="r2", return_train_score=True)
        grid.fit(trainDataX, trainDataY)
        best_alpha = grid.best_params_.get("alpha")
        regr = Lasso(alpha=best_alpha)
        regr.fit(trainDataX, trainDataY)
        pred = regr.predict(testDataX)
        print("Best alpha found to be %.5f" % best_alpha)
        print("Lasso, Grid Search Cross-validation - Residual square error: %.5f" % RSE(testDataY, pred))
        print("Lasso, Grid Search Cross-validation - Coefficient of determination: %.5f" % r2_score(testDataY, pred))
        displayGraph(xvals, grid.cv_results_.get("mean_test_score"), 'Lasso')

    # For Q1, validation portion   
    if (trainType == "validation"):
        SetX_train = trainDataX[:-int(trainDataSize/5)]
        SetX_test = trainDataX[-int(trainDataSize/5):]
        SetY_train = trainDataY[:-int(trainDataSize/5)]
        SetY_test = trainDataY[-int(trainDataSize/5):]
        regr.fit(SetX_train, SetY_train)
        SetY_pred = regr.predict(SetX_test)
        print("Validation - Residual square error: %.5f" % RSE(SetY_test, SetY_pred))
        print("Validation - Coefficient of determination: %.5f" % r2_score(SetY_test, SetY_pred))
    # For Q1, cross-validation portion
    elif (trainType == "cross-validation"):
        results = cross_validate(regr,
                        trainDataX,
                        trainDataY,
                        scoring=('r2', 'neg_mean_squared_error'),
                        cv=folds)
        r2s = results.get("test_r2")
        MSE = results.get("test_neg_mean_squared_error")
        RSE_total = 0
        r2s_total = 0
        for i in range(folds):
            RSE_total += math.sqrt((MSE[i]*-(trainDataSize/folds))/((trainDataSize/folds)- (params + 1)))
            r2s_total += r2s[i]
        print(f"Cross-validation, {folds}-folds - Residual square error: %.5f" % (RSE_total/folds))
        print(f"Cross-validation, {folds}-folds - Coefficient of determination: %.5f" % (r2s_total/folds))
    
    print("\n")

def Q1_results():
    print('Generating results for Q1...\n')
    
    Set = loadTrainData()
    SetY = Set['ConcreteCompressiveStrength_MPa_Megapascals_']
    SetX = Set.drop('ConcreteCompressiveStrength_MPa_Megapascals_', axis='columns')

    regression(SetX, SetY)
    regression(SetX, SetY, trainType="cross-validation")

def Q2_results():
    print('Generating results for Q2...\n')

    trainSet = loadTrainData()
    trainSetY = trainSet['ConcreteCompressiveStrength_MPa_Megapascals_']
    trainSetX = trainSet.drop('ConcreteCompressiveStrength_MPa_Megapascals_', axis='columns')

    testSet = loadTestData()
    testSetY = testSet['ConcreteCompressiveStrength_MPa_Megapascals_']
    testSetX = testSet.drop('ConcreteCompressiveStrength_MPa_Megapascals_', axis='columns')

    regression(trainSetX, trainSetY, testSetX, testSetY, model="ridge", trainType="gridcv")
    
def Q3_results():
    print('Generating results for Q3...\n')

    trainSet = loadTrainData()
    trainSetY = trainSet['ConcreteCompressiveStrength_MPa_Megapascals_']
    trainSetX = trainSet.drop('ConcreteCompressiveStrength_MPa_Megapascals_', axis='columns')

    testSet = loadTestData()
    testSetY = testSet['ConcreteCompressiveStrength_MPa_Megapascals_']
    testSetX = testSet.drop('ConcreteCompressiveStrength_MPa_Megapascals_', axis='columns')

    regression(trainSetX, trainSetY, testSetX, testSetY, model="lasso", trainType="gridcv")
    
###
#   Returns a vector of predictions of real number values,
#   corresponding to each of the N_test features vectors in Xtest
#   @params: Xtest N_test x 8 matrix of test feature vectors
#   @params: data_dir full path to the folder containing the following files: train.csv, test.csv
#   @return: N_test x 1 matrix of real number values
###
def predictCompressiveStrength(Xtest, data_dir):
    trainData = loadTrainData(data_dir)
    trainY = trainData['ConcreteCompressiveStrength_MPa_Megapascals_']
    trainX = trainData.drop('ConcreteCompressiveStrength_MPa_Megapascals_', axis='columns') 
    regr = Ridge(alpha= 5275)
    regr.fit(trainX, trainY)
    return regr.predict(Xtest)
    
#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
