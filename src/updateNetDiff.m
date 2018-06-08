function netDiff = updateNetDiff(netDiff, dataTrain, labelTrain)

    nrClasses = 9;
    
    % number of training examples
    dataSize = size(dataTrain, 2);
    
    % build true labels for training, that is:
    % for every label x:
    %   build a column vector with nrClasses(9) lines
    %   set the line x to 1
    
    trueLabelTrain = zeros(nrClasses, dataSize);
    indxLabel = 0 : dataSize - 1;
    indxLabel = nrClasses * indxLabel + labelTrain + 1;
    trueLabelTrain(indxLabel) = 1;

    netDiff = train(netDiff, dataTrain, trueLabelTrain);
end

