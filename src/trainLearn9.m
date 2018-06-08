function [bestNet, meanAcc, cMs] = trainLearn9(dataTrain, labelTrain)
   
    % number of training examples
    dataSize = size(dataTrain, 2);
    nrClasses = 2;
    
    % build nnet
    net9 = patternnet([75 37]);
    net9 = configure(net9, dataTrain, labelTrain);
    net9.layers{1}.transferFcn = 'tansig';
    net9.layers{2}.transferFcn = 'poslin';
    net9.layers{3}.transferFcn = 'logsig';
    
    net9.trainFcn = 'traingdx';
    net9.trainParam.epochs = 5000;
    net9.trainParam.goal = 1e-5;
    net9.trainParam.max_fail = 100;
    
    net9.divideParam.trainRatio = 0.80;
    net9.divideParam.valRatio = 0.20;
    net9.divideParam.testRatio = 0;
    
    % cross validation
    % divide dataTrain in 10 parts:
    % - at each step choose one of them as testing set
    % and use the other 9 remained for training
    
    maxPerf = 0;
    nrSubsets = 10;
    cMs = cell(1, nrSubsets);
    meanAcc = 0;
    
    nrExamplesSubset = dataSize / nrSubsets;
    perm = randperm(dataSize);

    dataSubsets = zeros(nrSubsets, nrExamplesSubset);
    dataSubsets(:) = perm;

    trainIndxs = zeros(1, (nrSubsets - 1) * nrExamplesSubset);
    
    for i = 1 : nrSubsets
        nrAdd = 0;
        for j = 1 : nrSubsets
            if j ~= i
                startPos = nrAdd * nrExamplesSubset + 1;
                endPos = startPos + nrExamplesSubset - 1;
                trainIndxs(startPos : endPos) = dataSubsets(j, :);
                nrAdd = nrAdd + 1;
            end
        end
        testIndxs = dataSubsets(i, :);

        % get the training and testing examples
        crossDataTrain = dataTrain(:, trainIndxs);
        crossLabelTrain = labelTrain(trainIndxs);

        crossDataTest = dataTrain(:, testIndxs);
        crossLabelTest = labelTrain(testIndxs);

        % train the network on the current training set
        net9 = init(net9);
        net9 = train(net9, crossDataTrain, crossLabelTrain);

        % evaluate the accuracy on the training set (perfTrain)
        % and on the test set (perTest)
        % the final score for the current network is calculated
        % by - perfTotal = 0.25 * perfTrain + 0.75 * perfTest;
        
        predictedLabels = (sim(net9, crossDataTrain) >= 0.5);
        perfTrain = length(find(predictedLabels == crossLabelTrain)) / length(crossLabelTrain);
        
        predictedLabels = (sim(net9, crossDataTest) >= 0.5);
        perfTest = length(find(predictedLabels == crossLabelTest)) / length(crossLabelTest);
        
        meanAcc = meanAcc + perfTest;
        % find confusion matrix for current testing data
        cMs{i} = confusionMatrix(crossLabelTest, predictedLabels, nrClasses);
        
        % calculate the final score for the current network
        perfTotal = 0.25 * perfTrain + 0.75 * perfTest;
        
        sprintf("%.3f %.3f %.3f", perfTrain, perfTest, perfTotal)
        
        % save the network with the highest perfTotal value
        if perfTotal > maxPerf
            maxPerf = perfTotal;
            bestNet = net9;
        end
    end
    
    meanAcc = meanAcc / nrSubsets;
    % at the end, train the chosen network(bestNet) on the
    % entire dataTrain, starting from the weights calculated 
    % in the cross-validation
    bestNet = train(bestNet, dataTrain, labelTrain);
end