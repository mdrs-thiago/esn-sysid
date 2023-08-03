function erro = RRSE(estimatedOutput, correctOutput)

  
nEstimatePoints = size(estimatedOutput, 1) ; 

nForgetPoints = size(correctOutput, 1) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 

meanCorrect = mean(correctOutput);

aux1 = sum(sum((estimatedOutput - correctOutput).^2)); 

aux2 = sum(sum((meanCorrect - correctOutput).^2));

erro = (sqrt(aux1/aux2)) ; 


