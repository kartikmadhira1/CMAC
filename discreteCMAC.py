#!/usr/bin/env python
'''
Homework #2 ENPM690
@author Kartik Madhira, UMD Robotics Graduate student
'''

import matplotlib.pyplot as plt
import numpy as np

nWeights = 35
learnRate = 0.03
testError = []
totalDatum = 100
trainDatum = 70
testDatum = 30

dataRange = totalDatum/10
x = np.arange(0, dataRange, 0.1)  
y = np.sin(x)

# Create train data in numpy frame
def getTrainData(x, y, dataRange):
	indTrainData = []
	trainData = []
	while(len(indTrainData) < trainDatum):
		indRandom = np.random.randint(100)
		if indRandom not in indTrainData:
			indTrainData.append(indRandom)
			trainData.append([x[indRandom], y[indRandom]])
	return trainData

# Create test data in numpy frame
def getTestData(x, y, dataRange, trainData):
	indTestData = []
	testData = []
	for values in range(totalDatum):
		if values not in trainData:
			indTestData.append(values)
			testData.append([x[values], y[values]])
	return testData

def orderedSet(set):
	# Get the train/test set in
	# bounds of 2.
	set = np.hsplit(set, 2)
	inputUnsorted = set[0]
	# Set out a temp variable to store 
	# intermittent values
	inputRandomSet = []
	for values in inputUnsorted:
		inputRandomSet.append(values[0])
	inputSorted = inputRandomSet
	inputSorted.sort()
	# Same way set out for output values
	outputSorted = set[1]
	outputRandomSet = []
	for values in outputSorted:
		outputRandomSet.append(values[0])
	# Same way set out intermittent values
	# for output.
	outputSorted = outputRandomSet
	# Save the sortedSet in a one-hot encoded style 
	# output
	sortedSet = []
	for values in inputSorted:
		index = np.where(inputUnsorted == values)
		sortedSet.append([values,outputSorted[index[0][0]]])
	return np.array(sortedSet)
	
trainData = np.array(getTrainData(x, y, dataRange))
testData = np.array(getTestData(x, y, dataRange, trainData))
print(trainData.shape, testData.shape)
trainData = orderedSet(trainData)


def trainContinuous(trainData, testData, dataVar, nEpochs):
	weightsFrame = []
	for values in range(nWeights):
		weightsFrame.append(float(np.random.randint(100))/100)
	epochErrorsTrain = []
	epochErrorsTest = []
	for eachEpoch in range(nEpochs+1):
		trainError = []
		predList = []
		for trainIndex in range(trainData.shape[0]):
			weightArray = []
			inputTrain = trainData[trainIndex][0]
			outputTrain = trainData[trainIndex][1]
			ratio = inputTrain/dataRange
			indWeightCenter = nWeights*ratio
			for indWeight in range(len(weightsFrame)):
				if indWeight < indWeightCenter + dataVar and indWeight > indWeightCenter - dataVar - 1:
					if indWeight < indWeightCenter - dataVar and indWeight >= indWeightCenter - dataVar - 1:
						weightArray.append(indWeight + 1 - (indWeightCenter - dataVar))
					elif indWeight <= indWeightCenter + dataVar and indWeight > indWeightCenter + dataVar - 1:
						weightArray.append(indWeightCenter + dataVar - indWeight)
					else:
						weightArray.append(1)
				else:
					weightArray.append(0)

			applyWeightsArr = []
			for indWeight in range(len(weightsFrame)):
				applyWeightsArr.append(weightArray[indWeight]*weightsFrame[indWeight])

			# Get a forward pass of the network
			output = 0
			for indWeight in range(len(weightsFrame)):
				output = output + applyWeightsArr[indWeight]*inputTrain

			#Error Calculation
			error = outputTrain - output
			trainError.append(-error)
			predList.append(output)
			weightsSum = sum(weightArray)
			for indWeight in range(len(weightsFrame)):
				weightsFrame[indWeight] = weightsFrame[indWeight] + error*learnRate*weightArray[indWeight]/weightsSum
		epochErrorsTrain.append(sum(trainError))


		# Test the data
		testError = []
		predList = []
		for indTest in range(testData.shape[0]):
			weightArray = []
			inputTest = testData[indTest][0]
			outputTest = testData[indTest][1]
			ratio = inputTest/dataRange
			indWeightCenter  = nWeights*ratio
			for indWeight in range(len(weightsFrame)):
				if indWeight < indWeightCenter + dataVar and indWeight > indWeightCenter - dataVar - 1:
					if indWeight<indWeightCenter -dataVar and indWeight >= indWeightCenter - dataVar - 1:
						weightArray.append(indWeight + 1 - (indWeightCenter - dataVar))
					elif  indWeight <= indWeightCenter + dataVar and indWeight > indWeightCenter + dataVar - 1:
						weightArray.append(indWeightCenter +dataVar-indWeight)
					else:
						weightArray.append(1)
				else:
					weightArray.append(0)

			applyWeightsArr = []
			for indWeight in range(len(weightsFrame)):
				applyWeightsArr.append(weightArray[indWeight]*weightsFrame[indWeight])

			# Get a forward pass of the network
			output = 0
			for indWeight in range(len(weightsFrame)):
				output = output + applyWeightsArr[indWeight]*inputTest
			error = outputTest - output
			testError.append(-error)
			predList.append(output)

		epochErrorsTest.append(sum(testError))

	print(np.mean(epochErrorsTest))
	return abs(np.mean(epochErrorsTest))

# iterate with overlaps from 1 to 34.
errList = []
for i in range(1,34):
	errList.append(trainContinuous(trainData, testData, i, i*100))

plt.plot(errList, color='c')
plt.xlabel('Overlap Area')
plt.ylabel('SMAPE')
plt.title('SMAPE vs Overlap Area ')
plt.savefig("conti_overlap_area_test.png")
plt.show()
