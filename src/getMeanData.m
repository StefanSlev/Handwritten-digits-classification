function dataTrain = getMeanData(dataTrain)
    
    % number of training examples
    dataSize = size(dataTrain, 2);
    
    % set all the NaN values in the dataTrain to 0
    % that's ok because there is no feature with value 0
    dataTrain(isnan(dataTrain)) = 0;
    
    % calculate the mean value of each feature (all the elements on one
    % line)
    meanRows = mean(dataTrain, 2);
    
    % set all the features with value 0(ex-NaN) to the mean of their corresponding
    % line
    for i = 1 : dataSize
        mask = (dataTrain(:, i) == 0);
        if sum(mask(:)) > 0
            dataTrain(:, i) = dataTrain(:, i) + mask .* meanRows;
        end
    end
end

