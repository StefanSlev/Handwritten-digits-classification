function cM = confusionMatrix(trueLabels, predictedLabels, nrClasses)
    
    cM = zeros(nrClasses);
    
    % build the confusion matrix from the trueLabels and the predicted ones
    for i = 1 : nrClasses
       for j = 1 : nrClasses
          trueLabel = i - 1;
          predictedLabel = j - 1;
          cM(i, j) = length(find(trueLabels == trueLabel & predictedLabels == predictedLabel));
       end
    end
end

