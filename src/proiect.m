load('dataTrain.mat');
load('labelTrain.mat');

dataDim = size(dataTrain, 1); % number of data features
dataSize = size(dataTrain, 2); % number of training examples
nrClasses = 10;

% remove the NaN values in the dataTrain
dataTrain = getMeanData(dataTrain);

% get the data and the labels for the examples
% with labels between 0 and 8
indxDiff = find(labelTrain ~= 9);
dataDiff = dataTrain(:, indxDiff);
labelDiff = labelTrain(indxDiff);

% train a binary classifier to distinguish 
% between data with label 9 and the rest of them

labelTrain9 = (labelTrain == 9);
[net9, meanAcc9, cMs9] = trainLearn9(dataTrain, labelTrain9);

% train a classifier to distinguish between
% data with labels in 0-8

[netDiff, meanAccDiff, cMsDiff] = trainLearnDiff(dataDiff, labelDiff);

% Find the predicted labels for the dataTrain by combining the 
% results of the two classifiers and evaluate their accuracy

predictedLabels = (sim(net9, dataTrain) >= 0.5);

% indx0 - examples that the net9 network
% classifies with non-9 label (0)

indx0 = find(predictedLabels == 0);

% find the labels (0 - 8) for the examples classified as non-9
% by net9

predictedLabelsDiff = sim(netDiff, dataTrain(:, indx0));
predictedOldLabelsDiff = getLabels(predictedLabelsDiff);

% combine the results
predictedLabels = predictedLabels * 9;
predictedLabels(indx0) = predictedOldLabelsDiff;

% calculate the accuracy
Perf = length(find(predictedLabels == labelTrain)) / length(labelTrain);
Perf

% Simulate networks on test data
load('dataTest.mat');

% remove the NaN values in the dataTest
dataTest = getMeanData(dataTest);

% Find the predicted labels for the dataTest
% as described above for the dataTrain

predictedLabels = (sim(net9, dataTest) >= 0.5);
indx0 = find(predictedLabels == 0);

predictedLabelsDiff = sim(netDiff, dataTest(:, indx0));
predictedOldLabelsDiff = getLabels(predictedLabelsDiff);

predictedLabels = predictedLabels * 9;
predictedLabels(indx0) = predictedOldLabelsDiff;

% Write the test labels in solution file
indx = 1 : size(dataTest, 2);
M = [double(indx)' double(predictedLabels)'];
csvwrite('solution.csv', M);