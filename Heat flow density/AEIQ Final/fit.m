function erro = fit(estimatedOutput, correctOutput)

  
nEstimatePoints = size(estimatedOutput, 1) ; 

nForgetPoints = size(correctOutput, 1) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 

%meanerror = sum((estimatedOutput - correctOutput).^2)/nEstimatePoints ; 
num = sqrt(sum((estimatedOutput-correctOutput).^2));
den = sqrt(sum((correctOutput-mean(correctOutput)).^2));
erro = (100*(1-num./den)); 

erro = mean(erro);