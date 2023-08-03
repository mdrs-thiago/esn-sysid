function erro = RMSE_mV(estimatedOutput, correctOutput)

  
nEstimatePoints = size(estimatedOutput, 1) ; 

nForgetPoints = size(correctOutput, 1) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 

meanerror = sum((estimatedOutput - correctOutput).^2)/nEstimatePoints ; 

erro = (sqrt(meanerror)*1000) ; 

erro = mean(erro);
