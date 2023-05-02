#
#  Assignment 1
#
#  Group 28:
#  <Brandon Cuza> <bhc107@mun.ca>

####################################################################################
# Imports
####################################################################################
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
def loadTrainData(dir_path=None):
    dir_path = dir_path or os.path.dirname(__file__)
    d0 = pd.read_csv(os.path.join(dir_path, 'train.sNC.csv'), header=None, names=['Feature 1', 'Feature 2'])
    d1 = pd.read_csv(os.path.join(dir_path, 'train.sDAT.csv'), header=None, names=['Feature 1', 'Feature 2'])
    d0['Outcome'] = 0
    d1['Outcome'] = 1
    trainingSet = pd.concat([d0, d1], ignore_index=True)
    return trainingSet

def loadTestData(dir_path=None):
    dir_path = dir_path or os.path.dirname(__file__)
    d0 = pd.read_csv(os.path.join(dir_path, 'test.sNC.csv'), header=None, names=['Feature 1', 'Feature 2'])
    d1 = pd.read_csv(os.path.join(dir_path, 'test.sDAT.csv'), header=None, names=['Feature 1', 'Feature 2'])
    d0['Outcome'] = 0
    d1['Outcome'] = 1
    testSet = pd.concat([d0, d1], ignore_index=True)
    return testSet

def classify(number_neigh, distance_metric):
    print('Performing classification...')
    
    trainingSet = loadTrainData()
    trainingSetOutcomes = trainingSet['Outcome']
    trainingSetFeatures = trainingSet.drop('Outcome', axis='columns')
    knnModel = KNeighborsClassifier(n_neighbors=number_neigh, metric=distance_metric)
    knnModel.fit(trainingSetFeatures.values, trainingSetOutcomes.values)
    
    grid = pd.read_csv(os.path.join(os.path.dirname(__file__), '2D_grid_points.csv'), header=None, names=['x', 'y'])
    grid_predict = knnModel.predict(grid.values)
    grid['Outcome'] = grid_predict

    testSet = loadTestData()
    testSetOutcomes = testSet['Outcome']
    testSetFeatures = testSet.drop('Outcome', axis='columns')

    trainingError = knnModel.score(trainingSetFeatures.values, trainingSetOutcomes.values)
    testError = knnModel.score(testSetFeatures.values, testSetOutcomes.values)
    
    colors = {0:'green', 1:'blue'}
    plt.title(f'k-Value = {number_neigh}, Metric = {distance_metric.capitalize()}\nTraining Error Rate = {round(1 - trainingError, 4)}, Test Error Rate = {round(1 - testError, 4)}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.scatter(grid['x'], grid['y'], color=grid['Outcome'].map(colors), marker='.')
    plt.scatter(trainingSet['Feature 1'], trainingSet['Feature 2'], color=trainingSet['Outcome'].map(colors))
    plt.scatter(testSet['Feature 1'], testSet['Feature 2'], color=testSet['Outcome'].map(colors), marker='+')
    plt.show()

def Q1_results():
    print('Generating results for Q1...')
    test = [1,3,5,10,20,30,50,100,150,200]
    for n in test:
        classify(n, 'euclidean')

def Q2_results():
    print('Generating results for Q2...')
    classify(30, 'manhattan')

def Q3_results():
    print('Generating results for Q3...')
    trainErrorDataX = []
    trainErrorDataY = []
    testErrorDataX = []
    testErrorDataY = []
    trainingSet = loadTrainData()
    trainingSetOutcomes = trainingSet['Outcome']
    trainingSetFeatures = trainingSet.drop('Outcome', axis='columns')
    testSet = loadTestData()
    testSetOutcomes = testSet['Outcome']
    testSetFeatures = testSet.drop('Outcome', axis='columns')
    for n in range(1,101):
        knnModel = KNeighborsClassifier(n_neighbors=n, metric='euclidean')
        knnModel.fit(trainingSetFeatures.values, trainingSetOutcomes.values)
        trainingAccuracy = knnModel.score(trainingSetFeatures.values, trainingSetOutcomes.values)
        trainingError = round(1 - trainingAccuracy, 4)
        testAccuracy = knnModel.score(testSetFeatures.values, testSetOutcomes.values)
        testError = round(1 - testAccuracy, 4)
        trainErrorDataX.append(1/n)
        testErrorDataX.append(1/n)
        trainErrorDataY.append(trainingError)
        testErrorDataY.append(testError)
    plt.title('Error rate versus Model capacity')
    plt.xlabel('Model capacity (1/k)')
    plt.ylabel('Error Rate')
    plt.xlim([0.01, 1.00])
    line1 = plt.plot(trainErrorDataX, trainErrorDataY, label='Training')
    line2 = plt.plot(testErrorDataX, testErrorDataY, label='Test')
    plt.legend()
    plt.xscale('log')
    plt.show()

def diagnoseDAT(Xtest, data_dir):

    trainSet = loadTrainData(data_dir)
    testSet = loadTestData(data_dir)
    
    totalData = pd.concat([trainSet, testSet], ignore_index=True)
    totalDataOutcomes = totalData['Outcome']
    totalDataFeatures = totalData.drop('Outcome', axis='columns')

    model = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
    model.fit(totalDataFeatures.values, totalDataOutcomes.values)

    modelPred = model.predict(Xtest)
    return modelPred

#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
