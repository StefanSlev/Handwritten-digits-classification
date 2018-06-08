function [bestNet, meanAcc, cMs] = trainLearnDiff(dataTrain, labelTrain)
   
    % number of training examples
    dataSize = size(dataTrain, 2);
    nrClasses = 9;
    
    % build nnet
    net = patternnet([45 45]);
    net.layers{1}.transferFcn = 'logsig';
    net.layers{2}.transferFcn = 'logsig';
    net.layers{3}.transferFcn = 'softmax';
    
    net.trainFcn = 'traingdx';
    net.trainParam.epochs = 5000;
    net.trainParam.goal = 1e-5;
    net.trainParam.max_fail = 25;
     
    net.divideParam.trainRatio = 0.80;
    net.divideParam.valRatio = 0.20;
    net.divideParam.testRatio = 0;
    
    % cross validation
    % divide dataTrain in 8 parts:
    % - at each step choose one of them as testing set
    % and use the other 7 remained for training
    
    maxPerf = 0;
    nrSubsets = 8;
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
        crossDataSize = size(crossDataTrain, 2);
        
        crossDataTest = dataTrain(:, testIndxs);
        crossLabelTest = labelTrain(testIndxs);

        % build true labels for training, that is:
        % for every label x:
        %   build a column vector with nrClasses(9) lines
        %   set the line x to 1
        
        trueLabelTrain = zeros(nrClasses, crossDataSize);
        indxLabel = 0 : crossDataSize - 1;
        indxLabel = nrClasses * indxLabel + crossLabelTrain + 1;
        trueLabelTrain(indxLabel) = 1;

        % train the network on the current training set
        net = init(net);
        net = train(net, crossDataTrain, trueLabelTrain);

        % evaluate the accuracy on the training set (perfTrain)
        % and on the test set (perTest)
        % the final score for the current network is calculated
        % by - perfTotal = 0.25 * perfTrain + 0.75 * perfTest;
        
        predictedLabels = sim(net, crossDataTrain);
        predictedOldLabels = getLabels(predictedLabels);
        perfTrain = length(find(predictedOldLabels == crossLabelTrain)) / length(crossLabelTrain);
        
        predictedLabels = sim(net, crossDataTest);
        predictedOldLabels = getLabels(predictedLabels);
        perfTest = length(find(predictedOldLabels == crossLabelTest)) / length(crossLabelTest);
        
        meanAcc = meanAcc + perfTest;
        % find confusion matrix for current testing data
        cMs{i} = confusionMatrix(crossLabelTest, predictedOldLabels, nrClasses);
        
        % calculate the final score for the current network
        perfTotal = 0.25 * perfTrain + 0.75 * perfTest;
        
        sprintf("%.3f %.3f %.3f", perfTrain, perfTest, perfTotal)
        
        % save the network with the highest perfTotal value
        if perfTotal > maxPerf
            maxPerf = perfTotal;
            bestNet = net;
        end
    end
    
    meanAcc = meanAcc / nrSubsets;
    % at the end, train the chosen network(bestNet) on the
    % entire dataTrain, starting from the weights calculated 
    % in the cross-validation
    bestNet = updateNetDiff(bestNet, dataTrain, labelTrain);
end