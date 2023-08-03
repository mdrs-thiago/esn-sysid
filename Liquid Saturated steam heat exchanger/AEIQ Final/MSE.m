function erro = MSE(estimatedOutput, correctOutput)

  
nEstimatePoints = size(estimatedOutput, 1) ; 

nForgetPoints = size(correctOutput, 1) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 

erro = sum((estimatedOutput - correctOutput).^2)/nEstimatePoints ; 


