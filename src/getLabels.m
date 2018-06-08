function labels = getLabels(matrixLabels)

    % number of training examples
    dataSize = size(matrixLabels, 2);
    
    labels = zeros(1, dataSize);
    
    %  - every column in matrixLabels is in fact a label for an example
    %  - this label x has on every line y a value between 0 and 1
    % meaning how high is the likelihood of that example to be in the class 
    % y - 1 (indexes start from 1 , classes from 0)
    
    % for every column label , get a single value = index of line with highest value - 1
    % representing the true label (value between 0 and 9) for that particular example
    
    for i = 1 : dataSize
        indxBest = find(matrixLabels(:, i) == max(matrixLabels(:, i)), 1);
        label = indxBest - 1;
        labels(i) = label;
    end
end

