function NRMSE = compute_NRMSE(estimatedOutput, correctOutput)

  
nEstimatePoints = size(estimatedOutput, 1) ; 

nForgetPoints = size(correctOutput, 1) - nEstimatePoints ; 

correctOutput = correctOutput(nForgetPoints+1:end,:) ; 

meanerror = sum((estimatedOutput - correctOutput).^2)/nEstimatePoints ; 

NRMSE = (sqrt(meanerror./var(correctOutput))) ; 