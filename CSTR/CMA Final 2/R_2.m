function erro = R_2(estimatedOutput, correctOutput)
  
nEstimatePoints = size(estimatedOutput, 1) ; 

nForgetPoints = size(correctOutput, 1) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 

c = cov(correctOutput,estimatedOutput);

erro = (c(1,2).^2)./(var(correctOutput).*var(estimatedOutput)); 


